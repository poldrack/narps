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
 scipy \
 seaborn \
 xlrd
RUN apt-get install -y vim

# install fsl
RUN apt-get install -y wget
RUN wget -O- http://neuro.debian.net/lists/stretch.us-nh.full |  tee /etc/apt/sources.list.d/neurodebian.sources.list
# ala https://github.com/inversepath/usbarmory-debian-base_image/issues/9
RUN mkdir ~/.gnupg
RUN echo "disable-ipv6" >> ~/.gnupg/dirmngr.confRUN apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9

RUN apt-get update && apt-get install -y --allow-unauthenticated fsl-core

CMD ["/bin/bash"]
