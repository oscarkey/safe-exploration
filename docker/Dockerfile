# NOTE: there is a bug with the latest torch version, resulting in a seg fault.
# Depending on which server you're running this on, you might need to modify the lines specifying cuda and pytorch versions.
# On some servers, that won't work either.
FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

### --build-args
ARG UID=1000

### CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64


RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         wget \
         sudo \
         vim \
         tmux \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         software-properties-common \
         && \
     rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends python3.7-dev

RUN curl --silent https://bootstrap.pypa.io/get-pip.py | python3.7

# Install some big dependencies before we copy the code in, to avoid having to install them every we update the code.
# TODO: there must be a better way to do this.
RUN pip install torch casadi matplotlib numpy scikit-learn scipy GPy gpytorch==0.3.2 pymongo dnspython pygame pytest \
                pytest-mock gym polytope jsonpickle
RUN pip install https://github.com/IDSIA/sacred/tarball/master#egg=sacred

# Section to get permissions right, and avoid running inside as root {{
# Create a user matching the UID, and create/chmod home dir (== project directory)
# (uid corresponds to grehar in CS network)
# <<UID>> will be replaced by user id
RUN useradd -d /project -u $UID --create-home user && echo "user:pass" | chpasswd && adduser user sudo

RUN mkdir /project/app
ADD . /project/app

RUN chown -R user:user /project

USER user
WORKDIR /project/app

RUN pip install --user -e ".[visualization,ssm_gpy,ssm_pytorch]"
RUN pip install --user -e ./constrained-cem-mpc

#ENV PYTHONPATH "$PYTHONPATH:/project/"