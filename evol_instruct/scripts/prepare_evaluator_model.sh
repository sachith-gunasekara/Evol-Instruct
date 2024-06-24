#!/bin/bash

while getopts "i:o:" opt
do
    case "$opt" in
        i) model_path_GGML="$OPTARG";;
        o) model_path_GGUF="$OPTARG";;
    esac
done


git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

if [ ! -f llama-cli ]; then
    echo "llama.cpp not compiled. Compiling now."
    make LLAMA_CUBLAS=1
fi

chmod +x llama-cli
echo "llama.cpp compiled."

cd ..

if [ -z "$model_path_GGML" ]; then
    if [ ! -z "$model_path_GGUF" ]; then
        echo "Cannot provide GGUF model path without corresponding GGML model path. Exiting."
        exit 1
    fi
else
    if [ -z "$model_path_GGUF" ]; then
        echo "Cannot provide GGML model path without corresponding GGUF model path. Exiting."
        exit 1
    else
        python3 llama.cpp/convert-llama-ggml-to-gguf.py -i $model_path_GGML -o $model_path_GGUF --eps 1e-5 --context-length 4096 --gqa 8
    fi
fi