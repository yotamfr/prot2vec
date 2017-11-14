#!/usr/bin/env bash

GPU_FLAG=''
#GPU_FLAG='-gpu'

PYTORCH_WHL='http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl'

virtualenv --system-site-packages -p python3.5 virtualenv

source virtualenv/bin/activate

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib6

easy_install -U pip

pip install $PYTORCH_WHL
pip install torchvision

pip3 install numpy

pip3 install --upgrade tensorflow$GPU_FLAG
pip3 install tf-nightly$GPU_FLAG
pip3 install tensorboard

pip3 install keras
git clone https://www.github.com/datalogai/recurrentshop.git
python recurrentshop/setup.py install

pip3 install biopython





