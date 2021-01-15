FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends apt-utils &&\
    apt-get -y install wget bc &&\
    apt-get -y install zip unzip &&\
    apt-get -y install libxt-dev &&\
    apt-get -y install libxext6 &&\
    apt-get -y install libglu1 &&\
    apt-get -y install libxrandr2 &&\
    apt-get -y install ghostscript &&\
    apt-get -y install imagemagick &&\
    apt-get -y install openjdk-8-jre &&\
    mkdir /tmp/matlab_mcr && \
    cd /tmp/matlab_mcr/ && \
    wget -nv http://www.mathworks.com/supportfiles/downloads/R2016a/deployment_files/R2016a/installers/glnxa64/MCR_R2016a_glnxa64_installer.zip && \
    unzip MCR_R2016a_glnxa64_installer.zip && \
    ./install -agreeToLicense yes -mode silent && \
    rm -rf /tmp/matlab_mcr

RUN mkdir /INPUTS && \
    mkdir /OUTPUTS && \
    mkdir /extra &&\
    mkdir /pythondir

ADD extra /extra


# Install Miniconda
RUN mkdir /tmp/miniconda &&\
    cd /tmp/miniconda &&\
    apt-get install bzip2 &&\
    wget -nv https://repo.continuum.io/miniconda/Miniconda2-4.3.30-Linux-x86_64.sh  --no-check-certificate &&\
    chmod +x Miniconda2-4.3.30-Linux-x86_64.sh &&\
    bash Miniconda2-4.3.30-Linux-x86_64.sh -b -p /pythondir/miniconda &&\
    rm -r /tmp/miniconda

# install Python packages
RUN /pythondir/miniconda/bin/pip install numpy -I &&\
    /pythondir/miniconda/bin/pip install numpy -U &&\
    /pythondir/miniconda/bin/pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl &&\
    /pythondir/miniconda/bin/pip install torchvision==0.2.0 &&\
    /pythondir/miniconda/bin/pip install pytz==2018.4 &&\
    /pythondir/miniconda/bin/pip install nibabel==2.2.1 &&\
    /pythondir/miniconda/bin/pip install tqdm==4.23.4 &&\
    /pythondir/miniconda/bin/pip install h5py==2.7.1 &&\
    /pythondir/miniconda/bin/pip install scipy==1.1.0 &&\
    /pythondir/miniconda/bin/pip install opencv-python==3.4.1.15 &&\
    /pythondir/miniconda/bin/pip install matplotlib==2.2.2 &&\
    /pythondir/miniconda/bin/pip install scikit-image==0.14.0


ENV PATH /pythondir/miniconda/bin:${PATH}
ENV CONDA_DEFAULT_ENV python27
ENV CONDA_PREFIX /pythondir/miniconda/envs/python27

#ENV LD_LIBRARY_PATH /usr/local/MATLAB/MATLAB_Runtime/v901/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v901/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v901/sys/os/glnxa64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/:/usr/local/MATLAB/MATLAB_Runtime/v901/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v901/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v901/sys/os/glnxa64:${LD_LIBRARY_PATH}
