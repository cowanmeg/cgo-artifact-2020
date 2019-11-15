# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,invalid-name
"""Bitserial conv2d schedule on arm cpu"""
from __future__ import absolute_import as _abs
import tvm
from tvm import autotvm
from .. import tag
from ..nn.pad import pad
from ..nn.bitserial_conv2d import bitserial_conv2d_nhwc, bitserial_conv2d_alter_layout
from ..nn.bitserial_util import bitpack, binary_op_multiplier
from ..nn.util import get_pad_tuple
from ..util import traverse_inline, get_const_int, get_const_tuple
from .. import generic

import os
from tvm.contrib import util, clang

def _kernel_vec_spatial_pack_nhwc(kernel, kernel_bits, VC, VCI, use_bitpack=True):
    if use_bitpack:
        kernel_q = bitpack(kernel, kernel_bits, pack_axis=2, bit_axis=2, pack_type='uint8')
    else:
        kernel_q = kernel
    KH, KW, KB, CI, CO = kernel_q.shape
    kvshape = (CO//VC, KH, KW, KB, VC, CI)
    return tvm.compute(kvshape, lambda co, dh, dw, b, vc, ci: \
        kernel_q[dh][dw][b][ci][co*VC+vc], name='kernel_vec')

@autotvm.register_topi_compute(bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def spatial_pack_nhwc(cfg, data, kernel, stride, padding, activation_bits, weight_bits,
                      pack_dtype, out_dtype, parallel):
    """ Compute convolution with pack on spatial axes. """
    assert data.shape[0].value == 1, "spatial pack convolution only support batch size=1"
    assert pack_dtype == 'uint8', "only support packing into uint8 bits"
    assert out_dtype == 'int16', "only support output type of int16"

    N, H, W, CI = get_const_tuple(data.shape)
    alter_kernel = True
    if len(kernel.shape) == 4:
        KH, KW, _, CO = get_const_tuple(kernel.shape)
        CI_packed = CI // 8
    elif len(kernel.shape) == 5:
        KH, KW, KB, CI_packed, CO = get_const_tuple(kernel.shape)
    else:
        OCO, KH, KW, KB, VC, CI_packed = get_const_tuple(kernel.shape)
        CO = OCO * VC
        alter_kernel = False
        kernel_vec = kernel
 
    if isinstance(padding, int) or (isinstance(padding, (tuple, list)) and len(padding) == 2):
        DPAD, RPAD, TPAD, LPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    PAD_H = H + (TPAD + DPAD)
    PAD_W = W + (LPAD + RPAD)
    OH = (PAD_H - KH) // HSTR + 1
    OW = (PAD_W - KW) // WSTR + 1
    oshape = (N, OH, OW, CO)

    # ==================== define configuration space ====================
    n, oh, ow, co = cfg.axis(N), cfg.axis(OH), cfg.axis(OW), cfg.axis(CO)
    ci, kh, kw = cfg.reduce_axis(CI_packed), cfg.reduce_axis(KH), cfg.reduce_axis(KW)
    ib, kb = cfg.reduce_axis(activation_bits), cfg.reduce_axis(weight_bits)

    oh, vh = cfg.define_split('tile_oh', oh, policy='all', num_outputs=2)
    ow, vw = cfg.define_split('tile_ow', ow, policy='all', num_outputs=2)
    
    # Microkernel constraints
    co, vc = cfg.define_split('tile_co', co, policy='all', num_outputs=2,
                              filter=lambda x: x.size[-1] == 8)
    ci_o, ci_i = cfg.define_split("tile_ci", ci, num_outputs=2,
                                filter=lambda x: x.size[-1] == 8 or x.size[-1] == 16)

    # Innermost 4 axis fixed for microkernel
    re_axes = cfg.define_reorder("reorder_0",
                                 [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                                 policy='candidate', candidate=[
                                     [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i],
                                     [n, oh, ow, co, vh, vw, kw, ci_o, kh, kb, ib, vc, ci_i],
                                     [n, oh, ow, co, vh, vw, ci_o, kw, kh, kb, ib, vc, ci_i]])
    # binary ops
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW * binary_op_multiplier(pack_dtype))
    # ====================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]
    VCI = cfg["tile_ci"].size[-1]

    data_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=3, pack_type='uint8')

    if alter_kernel:
        kernel_vec = _kernel_vec_spatial_pack_nhwc(kernel, weight_bits, VC, VCI, len(kernel.shape) == 4)

    N, H, W, IB, CI = data_q.shape
    dvshape = (N, OH // VH, OW // VW, VH*HSTR + KH-1, VW*WSTR + KW-1, IB, CI)
    ovshape = (1, OH // VH, OW // VW, CO // VC, VH, VW, VC)

    if (DPAD != 0 or TPAD != 0):
        data_pad = pad(data_q, (0, TPAD, LPAD, 0, 0), (0, DPAD, RPAD, 0, 0), name="data_pad")
    else:
        data_pad = data_q
    data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw,  b, ci: \
        data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][b][ci], name='data_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    dh = tvm.reduce_axis((0, KH), name='dh')
    dw = tvm.reduce_axis((0, KW), name='dw')
    ib = tvm.reduce_axis((0, IB), name='ib')
    kb = tvm.reduce_axis((0, KB), name='kb')

    def _unipolar_conv(n, h, w, co, vh, vw, vc):
        return tvm.sum(
            ((tvm.popcount(kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
                           data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci].astype('int16')) -
              tvm.popcount(~kernel_vec[co, dh, dw, kb, vc, ci].astype('int16') &
                           data_vec[n, h, w, vh*HSTR+dh, vw*WSTR+dw, ib, ci]).astype('int16'))
             << (kb + ib).astype('int16')), axis=[dh, dw, kb, ib, ci])

    tag = "parallel" if parallel else "single"
    conv_vec = tvm.compute(ovshape, _unipolar_conv, name='conv_vec', tag=tag)

    conv = tvm.compute(oshape, lambda n, h, w, co:
                       conv_vec[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC].astype(out_dtype),
                       name='conv', tag='spatial_bitserial_conv_nhwc')
    return conv

def _inline_ukernel(k):
    if k == 8:
        f = os.environ['ARTIFACT_HOME'] + "/ukernel-intrin.c"
    else:
        f = os.environ['ARTIFACT_HOME'] + "/ukernel-intrin-large.c"
    src = open(f).read()
    return clang.create_llvm(src, options=["-O3", "--target=armv7-none-linux-gnueabihf", "-mcpu=cortex-a53", "-mfpu=neon"])

def _intrin(m, k_i, w_b, x_b, unipolar):
    assert(m == 8)
    pack_dtype = 'uint8'
    w = tvm.placeholder((w_b, m, k_i), dtype=pack_dtype, name='w')
    x = tvm.placeholder((x_b, k_i,), dtype=pack_dtype, name='x')
    k = tvm.reduce_axis((0, k_i), name='k')
    bw = tvm.reduce_axis((0, w_b), name='bw')
    bx = tvm.reduce_axis((0, x_b), name='bx')
    if unipolar:
        dtype = 'int16'
        z = tvm.compute((m,), lambda i:
                        tvm.sum((tvm.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype)) -
                                 tvm.popcount(~w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype)))
                                << (bw+bx).astype(dtype), axis=[bw, bx, k]), name='z')
    else:
        dtype = 'uint16'
        z = tvm.compute((m,), lambda i:
                        tvm.sum(tvm.popcount(w[bw, i, k].astype(dtype) & x[bx, k].astype(dtype))
                                << (bw+bx).astype(dtype), axis=[bw, bx, k]), name='z')
    Ab = tvm.decl_buffer(w.shape, w.dtype,
                        name="A",
                        offset_factor=1,
                        data_alignment=k_i,
                        strides=[tvm.var("s1"), tvm.var("s2"), 1])
    Bb = tvm.decl_buffer(x.shape, x.dtype,
                        name="B",
                        offset_factor=1,
                        data_alignment=k_i,
                        strides=[tvm.var("si"), 1])

    Cb = tvm.decl_buffer(z.shape, z.dtype,
                        name="C",
                        offset_factor=1,
                        data_alignment=16,
                        strides=[1])
    def intrin_func(ins, outs):
        aa = ins[0]
        bb = ins[1]
        cc = outs[0]
        def _body():
            ib = tvm.ir_builder.create()
            if k_i == 8: 
                half = "_half"
            else:
                half = ""
            name = "update_unipolar_a%db%d%s" % (w_b, x_b, half)
            ib.emit(tvm.call_extern("int32", name,
                                aa.access_ptr("r"),
                                bb.access_ptr("r"),
                                cc.access_ptr("rw"), 
                                aa.strides[0], aa.strides[1], 
                                bb.strides[0]))

            return ib.get()
        def _reduce_reset():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", "reset",
                                cc.access_ptr("w")))
            return ib.get()
        def _reduce_update():
            return _body()
        return _body(), _reduce_reset(), _reduce_update()
    with tvm.build_config(offset_factor=1, partition_const_loop=True):
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds={w: Ab, x: Bb, z: Cb})

# ARM specific schedule that using custom microkernel
def _schedule_spatial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec, kernel_vec,
                                  conv_out, output, last, parallel):
    _, _, _, _, _, IB, CI = data_vec.shape
    _, KH, KW,  KB, _, _ = kernel_vec.shape
    KB = get_const_int(KB)
    IB = get_const_int(IB)
    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule data padding and  packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    s[data_q].compute_inline()

    _, h, w, _, _,  _, x = s[data_vec].op.axis
    fused = s[data_vec].fuse(h, w)
    if parallel:
        s[data_vec].parallel(fused)
    s[data_vec].vectorize(x)

    #### Schedule kernel packing
    if kernel_vec.op.name == 'kernel_vec':
        co, _, _, _, _, _ = s[kernel_vec].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[kernel_vec].pragma(co, 'debug_skip_region')
        elif parallel:
            s[kernel_vec].parallel(co)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    kh, kw, kb, ib, ci = s[conv_out].op.reduce_axis
    ci_o, ci_i = s[conv_out].split(ci, cfg['tile_ci'].size[1])
    re_axes = cfg["reorder_0"].apply(s, conv_out,
                                     [n, oh, ow, co, vh, vw, kh, kw, ci_o, kb, ib, vc, ci_i])

    # Use microkernel
    kfactor = cfg['tile_ci'].size[1]
    if kfactor % 8 == 0:
       pc = _intrin(VC, kfactor, KB, IB, True)
       s[conv_out].tensorize(kb, pc)
       s[conv_out].pragma(ci_o, "import_llvm", _inline_ukernel(kfactor))

    n, h, w, co = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)

    if last != output:
        s[output].compute_inline()

    s[conv_out].compute_at(s[last], co)
    oh_ow = s[last].fuse(oh, ow)
    s[last].vectorize(vc)
    if parallel:
        s[last].parallel(oh_ow)
    return s

@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nhwc, 'arm_cpu', 'direct')
def schedule_bitserial_conv2d_nhwc(cfg, outs):
    """Arm cpu schedule for bitserial conv2d"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        """Traverse operators from computation graph"""

        if 'spatial_bitserial_conv_nhwc' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[1]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data.op.input_tensors[0]
            parallel = "parallel" in conv_out.op.tag
            _schedule_spatial_conv2d_nhwc(cfg, s, data_q, data_pad, data_vec, kernel_vec,
                                          conv_out, output, outs[0], parallel)

    traverse_inline(s, outs[0].op, _callback)
    return s

##### REGISTER ALTER OP LAYOUT #####
@bitserial_conv2d_alter_layout.register(["arm_cpu"])
def _alter_bitserial_conv2d_layout_arm(attrs, inputs, tinfos, F):
    """Alter op layout for pre-computing kernel transformation

    Parameters
    ----------
    attrs : nnvm.top.AttrDict or tvm.attrs.Attrs
        Attributes of current convolution
    inputs : nnvm.symbol or tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    F: symbol
        The context, can be either nnvm.sym or relay.op

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level,
    so we have to pass 'F' to make it support our two versions of graph IR, NNVM and Relay.
    """
    copy_inputs = [s for s in inputs]

    new_attrs = {k: attrs[k] for k in attrs.keys()}

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    activation_bits = attrs.get_int("activation_bits")
    weight_bits = attrs.get_int("weight_bits")
    data_layout_key = "data_layout" if "data_layout" in new_attrs else "layout"
    layout = attrs[data_layout_key]
    out_dtype = attrs["out_dtype"]
    pack_dtype = attrs["pack_dtype"]
    unipolar = attrs["unipolar"]
    if out_dtype in ("same", ""):
        out_dtype = tinfos[0].dtype

    if layout == "NHWC":
        data, kernel = tinfos[0:2]
        N, H, W, CI = get_const_tuple(data.shape)
        if len(kernel.shape) == 4:
            KH, KW, _, CO = get_const_tuple(kernel.shape)
            CI_Packed = CI // 8
        else:
            KH, KW , _, CI_Packed, CO = get_const_tuple(kernel.shape)

        workload = autotvm.task.args_to_workload(
            [data, kernel, strides, padding, activation_bits, weight_bits,
            pack_dtype, out_dtype, unipolar], bitserial_conv2d_nhwc)

        target = tvm.target.current_target()
        dispatch_ctx = autotvm.DispatchContext.current
        cfg = dispatch_ctx.query(target, workload)
        
        if cfg.template_key == 'direct':  # pack weight tensor
            VC = cfg['tile_co'].size[-1]
            new_attrs['kernel_layout'] = 'OHWB%doI' % VC

            # Store the same config for the altered operator (workload)
            new_data = data
            new_kernel = tvm.placeholder((CO//VC, KH, KW, weight_bits, VC, CI_Packed), dtype=pack_dtype)
            new_workload = autotvm.task.args_to_workload(
                [new_data, new_kernel, strides, padding, activation_bits, weight_bits,
            pack_dtype, out_dtype, unipolar], bitserial_conv2d_nhwc)
            dispatch_ctx.update(target, new_workload, cfg)

            return F.nn.bitserial_conv2d(*copy_inputs, **new_attrs)
    
    else:
        return None
