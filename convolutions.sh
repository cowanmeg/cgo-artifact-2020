#!/bin/bash

# Measure layer by layer convolution performance for resnet18 on rasp3b
repeats=10
log_file=scripts/rasp3b.log

if [ ! -d data ]; then
  mkdir -p data;
fi

python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=1 \
        --activation_bits=1 --log_file=${log_file}
python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=1 \
         --activation_bits=2 --log_file=${log_file} 
python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=2 \
         --activation_bits=2  --log_file=${log_file}

# Extra convolutions for ablation - just need to benchmark the first convolution
python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=2 \
         --activation_bits=1  --log_file=${log_file} --first
python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=3 \
        --activation_bits=1  --log_file=${log_file} --first
python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=1 \
        --activation_bits=3  --log_file=${log_file} --first   

# A2W1 single threaded
 python3 scripts/convolutions_quantized.py --repeats=$repeats --weight_bits=1 \
         --activation_bits=2 --log_file=scripts/rasp3b_single.log --single

# Floating point
python3 scripts/convolutions_fp.py --repeats=$repeats


