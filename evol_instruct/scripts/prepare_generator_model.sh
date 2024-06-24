#!/bin/bash

while getopts "O:" opt
do
    case "$opt" in
        O) model_path="$OPTARG";;
    esac
done

git clone https://github.com/cmp-nct/ggllm.cpp
cd ggllm.cpp

if [ ! -f falcon_main ]; then
    echo "ggllm.cpp not compiled. Compiling now."

    export PATH="/usr/local/cuda/bin:$PATH"
    export LLAMA_CUBLAS=1
    
    make falcon_main falcon_quantize falcon_perplexity
fi

chmod +x falcon_main
echo "ggllm.cpp compiled."

cd ..

wget https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2/resolve/main/tokenizer.json -O "${model_path}/tokenizer.json"
