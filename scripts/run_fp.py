import os
import argparse
import numpy as np
import tvm
from tvm import autotvm, relay, rpc
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

autotvm.DispatchContext.current.silent = True

parser = argparse.ArgumentParser()
parser.add_argument('--repeats', type=int, default=10, help='number of repeats', required=False)
parser.add_argument('--local', action='store_true')
parser.set_defaults(local=False)
args = parser.parse_args()                                                                                                                                

repeats = args.repeats


input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)
dtype='float32'
net, r_params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1, dtype=dtype)

target = tvm.target.arm_cpu("rasp3b")
device_key = 'rpi3b'

def measure():
    print("Floating point")
    # compile kernels with tophub best records
    with autotvm.tophub.context(target):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
            net, target=target, params=r_params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # upload module to device
        host = os.environ['PI']
        port = int(os.environ['PORT'])
        remote = rpc.connect(host, port)
        ctx = remote.cpu()

        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=args.repeats, repeat=repeats)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        mean_ms = np.mean(prof_res)
        std_dev_ms = np.std(prof_res)
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (mean_ms, std_dev_ms))

        with open("data/end2end.csv", "a") as f:
            f.write("ARM,FP32,yes,%f,%f" % (mean_ms, std_dev_ms))
    

measure()