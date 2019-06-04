#!/usr/bin/env bash

# TODO with a little more effort, can download this over http using bazel
if [ ! -f "data/mnist.npz" ]; then
    echo "[databazel] Downloading data"
    mkdir -p data
    wget -P data https://s3.amazonaws.com/img-datasets/mnist.npz
fi

if [ ! -f "env-databazel-3/bin/python" ]; then
    echo "[databazel] Creating virtualenv and installing dependencies"
    virtualenv --python=python3 env-databazel-3
    ./env-databazel-3/bin/pip install -r requirements.txt
fi
