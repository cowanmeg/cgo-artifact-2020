# cgo-artifact-2020
Artifact repository for paper Automatic Generation of High-Performance Quantized Machine Learning Kernels

## Hardware Requirements:  
Host machine able to run Docker.   
Raspberry Pi 3B+ 

## Installation:

### Host machine:  

Download code and build docker image

`$ git clone https://github.com/cowanmeg/cgo-artifact-2020.git`

`$ docker build -t cgo-art cgo-artifact-2020/docker`

`$ docker run -it-v <absolute/path/cgo-artifact-2020>:/artifact cgo-art`

Build TVM

`$ cd artifact/tvm/build$ cmake .. && make -j4`

Copy scripts to raspberry pi 

`$ cd /artifact$ export PI=<Raspberry Pi's IP address>`

`$ scp -r pi <pi-user@ip>:/home/pi-user`

### Raspberry Pi:  

`$ cd /home/pi-user/pi`

`$ ./download_build.sh` 

## Experiments:

### Raspberry Pi:  
Benchmark handwritten quantized convolutions . 

`$ ./convolutions_pytorch.sh`

Start TVM RPC server - leave running for following experiments

`$ ./rpc.sh` 

Note: if port of server is not 9090 on the host machine run:

`$ export PORT=<port>`

### Host:
Synthesize ARM micro-kernels

`./synthesize.sh`

Benchmark quantized convolutions

`./convolutions.sh`

Benchmark ResNet18

`./end_to_end.sh`

Generate graphs

`./generate_graphs.sh`
