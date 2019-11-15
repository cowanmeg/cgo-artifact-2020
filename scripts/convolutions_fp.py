import os
import numpy as np
import tvm
from tvm import autotvm, rpc
from tvm.autotvm.task.space import FallbackConfigEntity
import topi
import topi.testing
from tvm.contrib import util
from topi.util import get_const_tuple
import argparse
from tvm.autotvm.task.topi_integration import serialize_args, TaskExtractEnv

parser = argparse.ArgumentParser()
parser.add_argument('--repeats', type=int, default=10, help='number of repeats when measuring time', required=False)
parser.set_defaults(local=False)
args = parser.parse_args()
args = parser.parse_args()


autotvm.GLOBAL_SCOPE.in_tuning = True
target = tvm.target.arm_cpu("rasp3b")
host = os.environ['PI']
port = int(os.environ['PORT'])
remote = rpc.connect(host, port)
ctx = remote.cpu()

data_file = 'data/tvm_conv2d_rasp.csv'
f = open(data_file, 'w') 


def upload_remote(func, name):
    # upload to rpi
    temp = util.tempdir()
    path = temp.relpath(name)
    func.save(path)
    remote.upload(path)
    return remote.load_module(name)

def get_ref_data(a_shape, w_shape, dtype, stride, padding):
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        c_np = topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding)
        return a_np, w_np, c_np

def verify_conv2d_nchw_winograd(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))

    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype
    a_np, w_np, c_np = get_ref_data(a_shape, w_shape, dtype, stride, padding)

    with target:
        with autotvm.tophub.context(target):

            C = topi.nn.conv2d(A, W, (stride, stride), (padding, padding), dilation=(1,1), layout='NCHW', out_dtype=dtype)
            s = topi.generic.schedule_conv2d_nchw([C])

    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)

    func = tvm.build(s, [A, W, C], target, name="conv_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))
    func = upload_remote(func, 'conv.o')

    func(a, w, c)
    if not autotvm.GLOBAL_SCOPE.in_tuning: 
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=args.repeats, repeat=args.repeats)
    results = evaluator(a, w, c).results
    mean_ms = np.mean(results) * 1000
    std_dev_ms = np.std(results) * 1000
    print(mean_ms, std_dev_ms, "ms")

    w = (batch, in_size, in_channel, num_filter, kernel, stride, padding)
    f.write(', '.join(str(x) for x in w))
    f.write(', ' + str(mean_ms) + ', ' + str(std_dev_ms) + '\n')

    return results


class WinogradFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'winograd'
        self.memory[key] = cfg
        print(cfg)
        return cfg

class DirectFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'direct'
        self.memory[key] = cfg
        print(cfg)
        return cfg


def verify_conv2d_nchw(batch, in_channel, in_size, num_filter, kernel, stride, padding):
    print("Workload: (%d, %d, %d, %d, %d, %d, %d)" % (batch, in_channel, in_size, num_filter, kernel, stride, padding))

    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    a_np, w_np, c_np = get_ref_data(a_shape, w_shape, dtype, stride, padding)

    with target:
        with autotvm.tophub.context(target):
            C = topi.nn.conv2d(A, W, (stride, stride), (padding, padding),
                                dilation=(1,1), layout='NCHW', out_dtype=dtype)
            s = topi.generic.schedule_conv2d_nchw([C])

            a = tvm.nd.array(a_np, ctx)
            w = tvm.nd.array(w_np, ctx)
            c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), ctx)


    func = tvm.build(s, [A, W, C], target, name="conv_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, ))
    func = upload_remote(func, 'conv.o')

    func(a, w, c)
    if not autotvm.GLOBAL_SCOPE.in_tuning: 
        tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=args.repeats, repeat=args.repeats)
    results = evaluator(a, w, c).results
    mean_ms = np.mean(results) * 1000
    std_dev_ms = np.std(results) * 1000
    print(mean_ms, std_dev_ms, "ms")

    w = (batch, in_size, in_channel, num_filter, kernel, stride, padding)
    f.write(', '.join(str(x) for x in w))
    f.write(', ' + str(mean_ms) + ', ' + str(std_dev_ms) + '\n')

    return results

def test_conv2d_nchw():
    # autotvm.DispatchContext.current.silent = True
    raw_data = 'data/raw_tvm_conv2d_fp_rasp3b.csv'

    all_results = np.empty((11, args.repeats))
    all_results[0] = verify_conv2d_nchw_winograd(1, 64, 56, 64, 3, 1, 1)
    all_results[1] = verify_conv2d_nchw(1,  64,  56,  64, 1, 1, 0)
    all_results[2] = verify_conv2d_nchw(1,  64,  56, 128, 3, 2, 1)
    all_results[3] = verify_conv2d_nchw(1,  64,  56, 128, 1, 2, 0)
    all_results[4] = verify_conv2d_nchw_winograd(1, 128, 28, 128, 3, 1, 1)
    all_results[5] = verify_conv2d_nchw(1, 128,  28, 256, 3, 2, 1)
    all_results[6] = verify_conv2d_nchw(1, 128,  28, 256, 1, 2, 0)
    all_results[7] = verify_conv2d_nchw_winograd(1, 256, 14, 256, 3, 1, 1)
    all_results[8] = verify_conv2d_nchw(1, 256,  14, 512, 3, 2, 1)
    all_results[9] = verify_conv2d_nchw(1, 256,  14, 512, 1, 2, 0)
    all_results[10] = verify_conv2d_nchw_winograd(1, 512, 7, 512, 3, 1, 1)

    np.savetxt(raw_data, all_results)

if __name__ == "__main__":
    TaskExtractEnv.get()
    test_conv2d_nchw()
