#!/usr/bin/env bash
set -e

# Activate the Python virtual environment
if [[ -f "env/bin/activate" ]]; then
    source env/bin/activate
else
    echo "Virtual environment not found. Exiting."
    exit 1
fi

#TODO add sampler.py pull from s3

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected."
    nvidia-smi

    # Download the model
    echo "Downloading Model"
    echo "$(pwd)"
    MODEL_URL="https://huggingface.co/TheBloke/CodeLlama-${PARAMS:-13}B-Instruct-GGUF/resolve/main/${MODEL:-codellama-13b-instruct.Q4_K_M}.gguf"
    if ! wget "$MODEL_URL"; then
        echo "Failed to download the model."
        exit 1
    fi

    # Install llama-cpp-python with specific CMake arguments
    CMAKE_ARGS="-DLLAMA_CUBLAS=on -DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.26 --no-cache-dir
else
    echo "NVIDIA GPU not detected or nvidia-smi not installed."
fi

# Execute passed commands
exec "$@"
