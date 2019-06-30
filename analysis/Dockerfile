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

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated fsl-core

# install R
RUN apt-get update && apt-get install -y default-jre gfortran
RUN wget https://cran.r-project.org/src/base/R-3/R-3.6.0.tar.gz
RUN tar zxf R-3.6.0.tar.gz
RUN cd R-3.6.0 && ./configure --enable-R-shlib=yes && make && make install
RUN echo 'install.packages(c("tidyverse","lme4"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R && rm -rf /workdir/R-3.6.0*
RUN echo 'install.packages(c("lmerTest","emmeans","pscl","arm","MuMIn"), repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R && rm -rf /R-3.6.0*
RUN apt-get install -y pandoc

ENV C_INCLUDE_PATH /usr/local/lib/R/include
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib
RUN echo "source /etc/fsl/5.0/fsl.sh" >> /root/.bashrc

RUN pip install statsmodels

CMD ["/bin/bash"]
