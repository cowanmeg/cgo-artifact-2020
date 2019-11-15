import os
import numpy as np
from PIL import Image
import argparse
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import riptide.anneal.models
    from riptide.anneal.models import get_model, get_optimizer
    from riptide.anneal.anneal_config import Config
    from riptide.anneal.anneal_funcs import *
    from riptide.utils.preprocessing.inception_preprocessing import preprocess_image
import tvm
from tvm import autotvm, relay, rpc
from tvm.relay import testing
import tvm.relay.testing.tf as tf_testing
from tvm.autotvm.tuner import XGBTuner, GATuner, GridSearchTuner
from tvm.contrib.util import tempdir
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.download import download_testdata
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

autotvm.DispatchContext.current.silent = True

parser = argparse.ArgumentParser()
parser.add_argument('--activation_bits', type=int, default=2, help='number of activation bits', required=False) 
parser.add_argument('--weight_bits', type=int, default=1, help='number of weight  bits', required=False) 
parser.add_argument('--repeats', type=int, default=10, help='number of repeats', required=False)
parser.add_argument('--log_file', type=str, default='log.log', help='logfile to store tuning results', required=False)
parser.add_argument('--local', action='store_true')
parser.add_argument('--nokernel', action='store_true')
parser.set_defaults(local=False)
parser.set_defaults(nokernel=False)
args = parser.parse_args()                                                                                                                                
activation_bits = args.activation_bits
weight_bits = args.weight_bits
repeats = args.repeats
log_file = args.log_file
target = tvm.target.arm_cpu("rasp3b")

os.environ["CUDA_VISIBLE_DEVICES"] = ''

# Checkpoints
if args.activation_bits == 2 and weight_bits == 1:
   checkpoint='scripts/checkpoints/resnet_a2w1'
elif activation_bits == 2 and weight_bits == 2:
   checkpoint='scripts/checkpoints/resnet_a2w2'
elif activation_bits == 1 and weight_bits == 1:
   checkpoint='scripts/checkpoints/resnet_a1w1'

config = Config(quantize=True, a_bits=float(activation_bits), w_bits=float(weight_bits), fixed=False)
with config:
    model = get_model("resnet18")
if args.nokernel:
    layout = "NCHW"
else:
    layout = "NHWC"

# Init model shapes.
test_input = tf.keras.Input(shape=[224, 224, 3], batch_size=1, dtype='float32')
output = model(test_input)

# Load checkpoint
ckpt = tf.train.latest_checkpoint(checkpoint)
parameters = tf.train.list_variables(ckpt)
gs = tf.compat.v1.train.get_or_create_global_step()
optimizer, learning_rate = get_optimizer('sgd', 0.1, gs, 64, 1)
ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
status = ckpt.restore(tf.train.latest_checkpoint(checkpoint))

pact_alphas = {}
sawb_scales = {}
tf_params = {}
for param in parameters:
    name = param[0]
    if 'pact' in name and 'alpha' in name and 'Momentum' not in name:
        key_name = name
        if 'unbinarized'in name:
            key_name = 'resnet18/pact/alpha' 
        pact_alphas[key_name] = float(tf.train.load_variable(checkpoint, name))
    elif 'sawb_conv2d' in name and 'Momentum' not in name:
        key_name = name
        fp_weights =  tf.train.load_variable(checkpoint, name)
        weights, scale = sawb_quantize(args.weight_bits, fp_weights)
        sawb_scales[name] = float(scale)
        tf_params[name] = weights.numpy().astype('int16')

# Parse model to tvm
with target:
    net, params = relay.frontend.from_keras(model, shape={'input_1': [1, 224, 224, 3]},
     weight_bits=args.weight_bits, activation_bits=args.activation_bits, layout=layout, 
     pact_alphas=pact_alphas, sawb_scales=sawb_scales, tf_params=tf_params)

def get_image(preprocess=True):
    img_path = 'scripts/dog.jpg'
    image = Image.open(img_path).resize((224, 224))
    if preprocess:
        data = preprocess_image(tf.constant(np.array(image)), 224, 224, is_training=False)
        data = data.numpy()
    else:
        data = np.asarray(image)
    data = data[np.newaxis, :]
    return data

# Evaluate performance.
def run_model():
    kernelstr = " no kernel" if args.nokernel  else ""
    print("A", args.activation_bits, "W", args.weight_bits, kernelstr, sep="")
    global net, params

    net = net[net.entry_func]
    # compile kernels with history best records.
    with autotvm.apply_history_best(log_file):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                net, target=target, params=params)

        # Upload module to device
        host = os.environ['PI']
        port = int(os.environ['PORT'])
        remote = rpc.connect(host, port)
        ctx = remote.cpu()

        # export library
        tmp = util.tempdir()
        lib_fname = tmp.relpath('net.tar')
        lib.export_library(lib_fname)

        # upload the library to remote device and load it
        remote.upload(lib_fname)
        rlib = remote.load_module('net.tar')

        # create the remote runtime module
        module = runtime.create(graph, rlib, ctx)

        # set parameter (upload params to the remote device. This may take a while)
        data = get_image()
        module.set_input(**params)

        synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                            '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                            '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                            'imagenet1000_clsid_to_human.txt'])
        synset_name = 'imagenet1000_clsid_to_human.txt'
        synset_path = download_testdata(synset_url, synset_name, module='data')
        with open(synset_path) as f:
            synset = eval(f.read())

        # Confirm correctness with tf model
        test_input = tf.constant(data.astype('float32'))
        output = model(test_input)  
        top1_tf = np.argmax(output[0].numpy())
        print('TF top-1 id: {}, class name: {}'.format(top1_tf, synset[top1_tf]))

        if args.nokernel:
             data = data.transpose((0, 3, 1, 2))
        module.set_input('input_1', data)
        module.run()
        tvm_out = module.get_output(0)
        top1_tvm = np.argmax(tvm_out.asnumpy()[0])
        print('RPI top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))

        # Check the actual vector output is within fp error
        np.testing.assert_allclose(output, tvm_out.asnumpy(), rtol=1e-3)

        # Benchmark time
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=repeats, repeat=args.repeats)
        prof_res = np.array(ftimer().results) * 1000 # Convert to milliseconds
        mean_ms = np.mean(prof_res)
        std_dev_ms = np.std(prof_res)
        print("Mean inference time (std dev): %.2f ms (%.2f ms)\n" %
                (mean_ms, std_dev_ms))

        with open("data/end2end.csv", "a") as f:
            ukernel = "no" if args.nokernel else "yes"
            f.write("ARM,A%dW%d,%s,%f,%f\n" % (args.activation_bits, 
                args.weight_bits, ukernel, mean_ms, std_dev_ms))
    
run_model()
