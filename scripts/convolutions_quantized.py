import os
import sys
import argparse
import numpy as np
import scipy.signal
import tvm
import topi
import topi.testing
from tvm import rpc, autotvm
from topi.util import get_const_tuple
from topi.nn.util import get_pad_tuple
from tvm.contrib import util
from tvm.autotvm.task.topi_integration import serialize_args, TaskExtractEnv

parser = argparse.ArgumentParser()
parser.add_argument('--activation_bits', type=int, default=1, help='number of activation bits', required=False)
parser.add_argument('--weight_bits', type=int, default=1, help='number of weight bits', required=False)
parser.add_argument('--repeats', type=int, default=10, help='number of repeats when measuring time', required=False)
parser.add_argument('--log_file', type=str, default=None, help='logfile to store tuning results', required=False)
parser.add_argument('--first', action='store_true', help='Only benchmark frist convolution')
parser.add_argument('--single', action='store_true', help='Disable parallelism')
parser.add_argument('--validate', action='store_true', help='Check correctness')
parser.set_defaults(local=False)
parser.set_defaults(single=False)
parser.set_defaults(validate=False)
args = parser.parse_args()
log_file = args.log_file

autotvm.GLOBAL_SCOPE.in_tuning = (not args.validate)

target = tvm.target.arm_cpu("rasp3b")
host = os.environ['PI']
port = int(os.environ['PORT'])
remote = rpc.connect(host, port)
ctx = remote.cpu()

def conv2d_nhwc_python(a_np, w_np, stride, padding):
    """Convolution operator in NHWC layout.
    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]
    w_np : numpy.ndarray
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]
    padding : np.ndarray
        4-D [top, left, bottom, right]
    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    batch, in_height, in_width, in_channel = a_np.shape
    kernel_h, kernel_w, _, num_filter = w_np.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    pad_top, pad_left, pad_bottom, pad_right = padding
    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_top + pad_bottom) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1
    # change the layout from NHWC to NCHW
    at = a_np.transpose((0, 3, 1, 2))
    wt = w_np.transpose((3, 2, 0, 1))
    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_top > 0 and pad_bottom > 0:
                    apad = np.zeros((in_height + pad_top + pad_bottom, 
                                     in_width + pad_left + pad_right))
                    apad[pad_top:-pad_bottom, pad_left:-pad_right] = at[n, c]
                elif pad_bottom > 0: # Asymmetric padding
                    apad = np.zeros((in_height + pad_top + pad_bottom, 
                                     in_width + pad_left + pad_right))
                    apad[:-pad_bottom, :-pad_right] = at[n, c]
                else:
                    apad = at[n, c]

                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(wt[f, c])), mode='valid')
                bt[n, f] += out[::stride_h, ::stride_w]
    return bt.transpose((0, 2, 3, 1))

def get_pad_pair(input1d, kernel1d, stride1d):
    out1d = (input1d + stride1d - 1) // stride1d
    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return [pad_before, pad_after]


def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)

def upload_remote(func, name):
    # upload to rpi
    temp = util.tempdir()
    path = temp.relpath(name)
    func.save(path)
    remote.upload(path)
    return remote.load_module(name)

# Verify implementation
def verify_bitserial_conv2d_nhwc(log_file, batch, in_size, in_channel, num_filter, kernel, stride, padding,
                        activation_bits, weight_bits, in_dtype, pack_dtype, out_dtype, parallel):
    in_height = in_width = in_size
    pad_before, pad_after = get_pad_pair(in_height, kernel, stride)
    pad_before = int(pad_before)
    pad_after = int(pad_after)
    pad = [pad_before, pad_before, pad_after, pad_after]
    stride = [stride, stride]


    with autotvm.apply_history_best(log_file):
        with target:
            A = tvm.placeholder((batch, in_height, in_width, in_channel), dtype=in_dtype, name='A')
            W = tvm.placeholder((kernel, kernel, in_channel, num_filter), dtype=in_dtype, name='W')
            WP = topi.nn.bitpack(W, weight_bits, 2, 2, 'uint8')
            WP2 = tvm.placeholder(WP.shape, WP.dtype, name='WP2')
            B = topi.nn.bitserial_conv2d_nhwc(A, WP2, stride, pad, activation_bits, weight_bits,
                                pack_dtype, out_dtype, parallel)
            sp = topi.generic.schedule_bitpack([WP])
            s = topi.generic.schedule_bitserial_conv2d_nhwc([B])

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), activation_bits, in_dtype)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, in_dtype)
        w_ = np.copy(w_np).astype(out_dtype)
        # Need to map the bipolar weights to non-negative integers
        if weight_bits == 1:
            # -1, 1 --> 0, 1
            for x in np.nditer(w_, op_flags=['readwrite']):
                x[...] = 1 if x == 1 else -1
            # -3, -1, 1, 3 --> 0, 1, 2, 3
        elif weight_bits == 2:
            for x in np.nditer(w_, op_flags=['readwrite']):
                if x == 0:
                    x[...] = -3
                elif x == 1:
                    x[...] = -1
                elif x == 2:
                    x[...] = 1
                elif x == 3:
                    x[...] = 3
                else:
                    assert True
        elif weight_bits == 3:
            for x in np.nditer(w_, op_flags=['readwrite']):
                if x == 0:
                    x[...] = -7
                elif x == 1:
                    x[...] = -5
                elif x == 2:
                    x[...] = -3
                elif x == 3:
                    x[...] = -1
                elif x == 4:
                    x[...] = 1
                elif x == 5:
                    x[...] = 3
                elif x == 6:
                    x[...] = 5
                elif x == 7:
                    x[...] = 7
                else:
                    assert True
        elif weight_bits == 4:
            for x in np.nditer(w_, op_flags=['readwrite']):
                if x == 0:
                    x[...] = -15
                elif x == 1:
                    x[...] = -13
                elif x == 2:
                    x[...] = -11
                elif x == 3:
                    x[...] = -9
                elif x == 4:
                    x[...] = -7
                elif x == 5:
                    x[...] = -5
                elif x == 6:
                    x[...] = 3
                elif x == 7:
                    x[...] = -3
                elif x == 8:
                    x[...] = -1
                if x == 9:
                    x[...] = 1
                elif x == 10:
                    x[...] = 3
                elif x == 11:
                    x[...] = 5
                elif x == 12:
                    x[...] = 7
                elif x == 13:
                    x[...] = 9
                elif x == 14:
                    x[...] = 13
                elif x == 15:
                    x[...] = 15
                else:
                    assert True
        else:
            assert True, "Add mapping for weight bits > 2"
        b_np = conv2d_nhwc_python(a_np, w_, stride, pad).astype(out_dtype)
      
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()
    
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    wp = tvm.nd.array(np.zeros(get_const_tuple(WP.shape), dtype=WP.dtype), ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)

    func1 = tvm.build(sp, [W, WP], target)
    func = tvm.build(s, [A, WP2, B], target)

    func = upload_remote(func, 'bs_conv.o')
    func1 = upload_remote(func1, 'bitpack.o')

    func1(w, wp)
    func(a, wp, b)

    if not autotvm.GLOBAL_SCOPE.in_tuning:
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, verbose=True)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=args.repeats, repeat=args.repeats)
    results = evaluator(a, wp, b).results
    return results


def test_bitserial_conv2d_nhwc(workloads):
    parallel = not args.single
    if parallel:
        raw_data = 'data/raw_bitserial_conv2d_nhwc_a%dw%d_rasp3b.csv' % (args.activation_bits, args.weight_bits)
    else:
        raw_data = 'data/raw_bitserial_conv2d_nhwc_a%dw%d_rasp3b_single.csv' % (args.activation_bits, args.weight_bits)

    activation_bits = args.activation_bits
    weight_bits = args.weight_bits
    in_dtype = 'int8'
    pack_dtype = 'uint8'
    out_dtype = 'int16'

    # Measure and record times
    all_results = np.empty((len(workloads), args.repeats))
    for itr, w in enumerate(workloads):
        (batch, in_size, ic, oc, k, stride, padding) = w
        results = verify_bitserial_conv2d_nhwc(log_file, batch, in_size, ic, oc, k, 
            stride, padding, activation_bits, weight_bits, in_dtype, 
            pack_dtype, out_dtype, parallel)
        mean_ms = np.mean(results) * 1000
        std_dev_ms = np.std(results) * 1000
        print("Workload", w, " Average time", mean_ms, "ms", "std deviation", std_dev_ms, "ms")
        all_results[itr] = results
    np.savetxt(raw_data, all_results)

if __name__ == "__main__":
    # Convolutions layers of resent minus the first which is traditionally not binarized
    TaskExtractEnv.get()
    resnet_workload = [
        (1, 56, 64, 64, 3, 1, 'SAME'), 
        (1, 56, 64, 64, 1, 1, 'VALID'),

        (1, 56, 64, 128, 3, 2, 'SAME'),
        (1, 56, 64, 128, 1, 2, 'VALID'),
        (1, 28, 128, 128, 3, 1, 'SAME'),

        (1, 28, 128, 256, 3, 2, 'SAME'),
        (1, 28, 128, 256, 1, 2, 'VALID'),
        (1, 14, 256, 256, 3, 1, 'SAME'),

        (1, 14, 256, 512, 3, 2, 'SAME'),
        (1, 14, 256, 512, 1, 2, 'VALID'),
        (1, 7, 512, 512, 3, 1, 'SAME')
         ]
    if args.first:
        resnet_workload = resnet_workload = [resnet_workload[0]]

    print("A", args.activation_bits, "W", args.weight_bits, sep='')
    test_bitserial_conv2d_nhwc(resnet_workload)
