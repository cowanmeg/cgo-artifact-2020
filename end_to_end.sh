#!/bin/bash

# Runs normal graph runtime to get end to end numbers on local raspberry pi

repeats=10
log_file=scripts/rasp3b.log
output=data/end2end.csv

if [ ! -d data ]; then
  mkdir -p data;
fi

rm -f ${output}
touch ${output}
echo "arch,type,kernel,avg-ms,std-dev-ms" >> ${output}

# Quantized with microkernel
python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=1 --weight_bits=1  --log_file=${log_file}
python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=2 --weight_bits=1  --log_file=${log_file}
python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=2 --weight_bits=2 --log_file=${log_file}

 # # Quantized without microkernel 
 log_file=scripts/rasp3b_nchw.log
 python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=1 --weight_bits=1  --log_file=${log_file} --nokernel
 python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=2 --weight_bits=1  --log_file=${log_file} --nokernel
 python3 scripts/run_quantized.py --repeats=$repeats --activation_bits=2 --weight_bits=2 --log_file=${log_file} --nokernel

 # Floating point
 python3 scripts/run_fp.py --repeats=$repeats 
