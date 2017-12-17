#!/usr/bin/env bash

CONDA_URL="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash Miniconda3-latest-Linux-x86_64.sh

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

### Prerequisites

#sudo apt-get install blast2
#sudo apt-get install ncbi-blast+

#wget http://raptorx.uchicago.edu/download/eHVlZmVuZy5jdWlAa2F1c3QuZWR1LnNh/25/ #nr90.tar.gz


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

pip install git+https://github.com/RaRe-Technologies/gensim.git # gensim with poincare emb

pip install pymongo

pip install tqdm

pip install networkx

pip install obonet

### install virtualenv as kernel to ipython notebook
pip install ipykernel
python -m ipykernel install --user --name virtualenv

#git clone https://www.github.com/datalogai/recurrentshop.git
#python recurrentshop/setup.py install

#git clone https://github.com/aditya-grover/node2vec.git









