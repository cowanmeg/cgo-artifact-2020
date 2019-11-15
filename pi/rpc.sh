export PYTHONPATH=$(pwd)/tvm/python
python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090