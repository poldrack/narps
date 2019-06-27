# Dockerfile for ThinkStats

FROM python:3.6-stretch
MAINTAINER Russ Poldrack <poldrack@gmail.com>


RUN apt-get clean all
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get autoremove
RUN apt-get install gcc

RUN pip install \
 nibabel \
 nipype \
 nilearn 
RUN pip install jupyter
RUN pip install numpy \
 pandas \
 scikit-learn \
 matplotlib \
 scipy\
 seaborn
RUN apt-get install -y vim

CMD ["/bin/bash"]
