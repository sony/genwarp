FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG USER_NAME=genwarp
ARG GROUP_NAME=genwarp
ARG UID=1000
ARG GID=1000

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

RUN apt update

# Avoid timezone cofiguration.
RUN DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    apt install -y tzdata

# Install packages.
RUN apt install -y \
    libnvidia-gl-535 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglm-dev \
    mesa-utils

RUN apt update && apt install -y \
    git \
    curl \
    ffmpeg \
    software-properties-common

# Python 3.10.
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update && apt install -y \
    python3.10 python3.10-dev
RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

# Alias.
RUN ln -s $(command -v python3.10) /usr/bin/python

# Change user to non-root user.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/bash -u ${UID} -g ${GID} ${USER_NAME}
USER ${USER_NAME}

# Install dependencies.
RUN pip install --upgrade pip setuptools==69.5.1 ninja
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118
