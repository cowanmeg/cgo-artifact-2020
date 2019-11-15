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
"""Bitserial conv2d schedule on x86"""
import tvm
from tvm import autotvm
from topi.util import get_const_int, traverse_inline
from .. import generic, tag

@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nchw, ['cpu'], 'direct')
@autotvm.register_topi_schedule(generic.nn.schedule_bitserial_conv2d_nhwc, ['cpu'], 'direct')
def schedule_bitserial_conv2d(cfg, outs):
    """CPU schedule for bitserial convolutions NCHW and NHWC"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        """Traverse operators from computation graph"""

        if 'spatial_bitserial_conv' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_q = kernel_vec.op.input_tensors[0]
            data_vec = conv_out.op.input_tensors[0]
            data_q = data_vec.op.input_tensors[0]
            data = data_q.op.input_tensors[0]
            data_pad = None
            if isinstance(data_q.op, tvm.tensor.ComputeOp) and "pad" in data_q.op.tag:
                data_pad = data_q
                data_q = data
                data = data.op.input_tensors[0]
            if 'nhwc' in op.tag:
                _schedule_bitserial_conv2d_nhwc(cfg, s, data_pad, data_vec, kernel_vec,
                                            conv_out, output, outs[0])
            else:
                _schedule_bitserial_conv2d_nchw(cfg, s, data_pad, data_vec, kernel_vec,
                                            conv_out, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

def _schedule_bitserial_conv2d_nchw(cfg, s, data_pad, data_vec, kernel_vec,
                                    conv_out, output, last):

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]
     
    ##### Schedule Data padding, and bitpacking
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    s[data_vec].parallel(h)

    ##### Schedule Kenerl bitpacking
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    s[kernel_vec].parallel(co)

   ##### Schedule Convolution
    n, co, oh, ow, vh, vc, vw = s[conv_out].op.axis
    ci, kh, kw, ib, kb = s[conv_out].op.reduce_axis
    cfg["reorder_0"].apply(s, conv_out, [n, co, oh, ow, kh, kw, kb, ib, ci, vh, vw, vc])
    cfg["ann_reduce"].apply(s, conv_out, [kb, ib, kh, kw],
                            axis_lens=[get_const_int(kb.dom.extent),
                                       get_const_int(ib.dom.extent),
                                       get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv_out, [vh, vw, vc],
                             axis_lens=[VH, VW, VC],
                             max_unroll=16,
                             cfg=cfg)

    # # Schedule output
    n, co, h, w = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, co, oh, ow, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=16,
                                 cfg=cfg)
    s[conv_out].compute_at(s[last], ow)

    s[last].parallel(co)
    return s

def _schedule_bitserial_conv2d_nhwc(cfg, s, data_pad, data_vec, kernel_vec,
                                    conv_out, output, last):
    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    ##### Schedule data padding and packing
    if data_pad is not None:
        s[data_pad].compute_inline()

    _, h, _, _, _, _, _ = s[data_vec].op.axis
    cfg.define_split("tile_ah", cfg.axis(h), policy="all", num_outputs=2, max_factor=32)
    oh, ih = cfg["tile_ah"].apply(s, data_vec, h)
    s[data_vec].parallel(oh)

    ##### Schedule kernel packing
    co, _, _, _, _, _ = s[kernel_vec].op.axis
    cfg.define_split("tile_bco", cfg.axis(co), policy="all", num_outputs=2, max_factor=32)
    oco, ico = cfg["tile_bco"].apply(s, kernel_vec, co)
    s[kernel_vec].parallel(oco)

    ##### Schedule Convolution
    n, oh, ow, co, vh, vw, vc = s[conv_out].op.axis
    dh, dw, ci, b1, b2 = s[conv_out].op.reduce_axis

    # s[conv_out].reorder(n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2)
    cfg["reorder_0"].apply(s, conv_out, [n, oh, ow, co, vh, vw, dh, dw, ci, vc, b1, b2])
    cfg["ann_reduce"].apply(s, conv_out, [b1, b2, dh, dw],
                            axis_lens=[get_const_int(b1.dom.extent),
                                       get_const_int(b2.dom.extent),
                                       get_const_int(dh.dom.extent),
                                       get_const_int(dw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)

    s[conv_out].unroll(b1)
    s[conv_out].unroll(b2)
    s[conv_out].vectorize(vc)

    # # Schedule output
    n, h, w, co = s[last].op.axis
    co, vc = s[last].split(co, VC)
    oh, ow, vh, vw = s[last].tile(h, w, VH, VW)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)
    s[last].vectorize(vc)
    if last != output:
        s[output].compute_inline()
    s[conv_out].compute_at(s[last], ow)

    oho, iho = cfg["tile_oh"].apply(s, last, oh)  # reuse parameter
    s[last].parallel(oho)

    return s
