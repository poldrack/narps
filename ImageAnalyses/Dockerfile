# Dockerfile for ThinkStats

FROM python:3.6-stretch

# apt-get installs

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc=4:6.3.0-4 \
    vim=2:8.0.0197-4+deb9u3 \
    wget=1.18-5+deb9u3 \
    default-jre=2:1.8-58+deb9u1 \
    gfortran=4:6.3.0-4 \
    pandoc=1.17.2~dfsg-3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip installs

RUN pip install \
    nibabel==2.4.1 \
    nipype==1.2.0 \
    nilearn==0.5.2 \
    jupyter==1.0.0 \
    numpy==1.16.4 \
    pandas==0.24.2 \
    scikit-learn==0.21.2 \
    matplotlib==3.1.0 \
    scipy==1.2.1 \
    seaborn==0.9.0 \
    xlrd==1.2.0 \
    statsmodels==0.10.0 \
    wget==3.2 \
    requests==2.22.0 \
    Pillow==5.3.0 \
    extract==1.0 \
    python-dotenv==0.10.3 \
    nipy==0.4.2 \
    colorama==0.4.1
    

RUN pip install git+https://github.com/poldrack/neurovault_collection_downloader.git@f733fd88460230fc39b815574050082f319d7cb7

RUN pip install git+https://github.com/neurostuff/NiMARE.git@0170c712308742442e0b32b5f4fedd949ba6358e#egg=nimare[peaks2maps-cpu]

# install fsl
RUN wget -O- http://neuro.debian.net/lists/stretch.us-nh.full |  tee /etc/apt/sources.list.d/neurodebian.sources.list
# ala https://github.com/inversepath/usbarmory-debian-base_image/issues/9
RUN mkdir ~/.gnupg
RUN echo "disable-ipv6" >> ~/.gnupg/dirmngr.confRUN apt-key adv --recv-keys --keyserver hkp://pool.sks-keyservers.net:80 0xA5D32F012649A5A9
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --allow-unauthenticated fsl-core=5.0.9-5~nd90+1 fsl-mni152-templates=5.0.7-2 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install R and necessary packages
RUN wget https://cran.r-project.org/src/base/R-3/R-3.6.0.tar.gz && tar zxf R-3.6.0.tar.gz -C /
WORKDIR /R-3.6.0
RUN ./configure --enable-R-shlib=yes && make && make install
RUN rm -rf /R-3.6.0*

RUN echo 'install.packages(c("checkpoint"), repos="http://cran.us.r-project.org", dependencies=TRUE)' >> /tmp/packages.R && Rscript /tmp/packages.R && rm -rf /R-3.6.0*

# environment setup
ENV C_INCLUDE_PATH /usr/local/lib/R/include
ENV LD_LIBRARY_PATH /usr/local/lib/R/lib
RUN echo "source /etc/fsl/5.0/fsl.sh" >> /root/.bashrc
ENV NARPS_BASEDIR /data

WORKDIR /analysis

CMD ["/bin/bash"]
