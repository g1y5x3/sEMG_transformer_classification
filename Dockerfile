FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y &&\
    apt-get install -y vim  && \
    apt-get install -y git  && \
    pip install wandb && \
    pip install scikit_learn && \
    pip install sktime && \
    pip install numpy && \
    pip install scipy && \
    pip install pandas && \
    pip install matplotlib && \
    pip install tqdm && \
    pip install ipdb && \
    pip install xlrd && \
    pip install xlutils && \
    pip install tabulate && \
    pip install xlwt && \
    pip install tensorboard

RUN git config --global user.name  "Yixiang Gao" && \
    git config --global user.email "yg5d6@umsystem.edu"

ADD data/Multivariate_ts/EthanolConcentration /home/data
ADD data/subjects_40_vowels_v6.mat /home/data/