#!/bin/bash

git clone https://github.com/cmp-nct/ggllm.cpp
cd ggllm.cpp

rm -rf build; mkdir build; cd build

export PATH="/usr/local/cuda/bin:$PATH"
export LLAMA_CUBLAS=1

make falcon_main falcon_quantize falcon_perplexity
cd ..