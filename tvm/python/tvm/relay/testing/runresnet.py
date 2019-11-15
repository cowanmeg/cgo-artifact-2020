import tvm
from tvm import relay
from tvm.relay import testing
import tvm.contrib.graph_runtime as runtime

net, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1, dtype='float32')
print(net)
