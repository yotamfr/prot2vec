#!/usr/bin/env bash

GPU_FLAG=""

while getopts ":g:" opt; do
  case $opt in
    g)
    ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "-gpu was triggered."
      GPU_FLAG='-gpu'
      ;;
  esac
done


conda create -p virtualenv python=3.6

conda install -p virtualenv pip -c anaconda


if [ -z $GPU_FLAG ] ; then
    conda install -p virtualenv pytorch torchvision -c soumith

else
    conda install -p virtualenv pytorch torchvision cuda80 -c soumith
fi

conda install -p virtualenv visdom -c conda-forge


### activate virtualenv
source activate virtualenv

### use pip to install remaining packages
pip install --upgrade tensorflow$GPU_FLAG
pip install tf-nightly$GPU_FLAG
pip install keras

pip  install scikit-learn

pip install matplotlib

pip install wget    # Does NOT install properly by conda

pip install pandas

pip install biopython

pip install gensim

pip install pymongo

pip install tqdm

### install virtualenv as kernel to ipython notebook
pip install ipykernel
python -m ipykernel install --user --name virtualenv

git clone https://www.github.com/datalogai/recurrentshop.git
python recurrentshop/setup.py install






