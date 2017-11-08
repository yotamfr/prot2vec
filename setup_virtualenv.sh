#!/usr/bin/env bash

WHL = http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl

virtualenv --system-site-packages -p python3.5 virtualenv

source virtualenv/bin/activate

easy_install -U pip

pip3 install $WHL
pip3 install torchvision

