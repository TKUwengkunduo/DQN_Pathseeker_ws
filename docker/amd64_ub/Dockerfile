# 使用 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 基礎映像檔
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 
# CUDA11.2, cudnn8.1.1

# Maintainer Information
LABEL maintainer="wengkunduo@gmail.com"

RUN apt-get update

# 關閉互動模式
ENV DEBIAN_FRONTEND=noninteractive

## NVIDIA GraphicsCard parameter
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# 接收外部參數
ARG USER=initial
ARG UID=1000
ARG GROUP=initial
ARG GID=1000

# 創建一個名為initial的用戶和用戶組
RUN groupadd -g $GID $GROUP && \
    useradd -m -u $UID -g $GID -s /bin/bash $USER

# Update and Install Packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    at-spi2-core \
    htop git wget curl \
    byobu zsh \
    terminator \
    dbus-x11 libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 libxtst6 libinih-dev\
    nano vim \
    gnome-terminal libcanberra-gtk-module libcanberra-gtk3-module \
    python3-wheel python3-pip python3-dev python3-setuptools \
    libopencv-dev \
    openssh-server \
    cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安裝opencv與tensorflows
RUN pip3 install opencv-python tensorflow==2.11.0 matplotlib
RUN pip3 install matplotlib gym tqdm

# Tensorflow GPU https://www.tensorflow.org/install/source#linux
# Version	            Python version	Compiler	    Build tools	    cuDNN	CUDA
# tensorflow-2.14.0	    3.9-3.11	    Clang 16.0.0	Bazel 6.1.0	    8.7	    11.8
# tensorflow-2.13.0	    3.8-3.11	    Clang 16.0.0	Bazel 5.3.0	    8.6	    11.8
# tensorflow-2.12.0	    3.8-3.11	    GCC 9.3.1	    Bazel 5.3.0	    8.6	    11.8
# tensorflow-2.11.0	    3.7-3.10	    GCC 9.3.1	    Bazel 5.3.0	    8.1	    11.2
# tensorflow-2.10.0	    3.7-3.10	    GCC 9.3.1	    Bazel 5.1.1	    8.1	    11.2
# tensorflow-2.9.0	    3.7-3.10	    GCC 9.3.1	    Bazel 5.0.0	    8.1	    11.2
# tensorflow-2.8.0	    3.7-3.10	    GCC 7.3.1	    Bazel 4.2.1	    8.1	    11.2
# tensorflow-2.7.0	    3.7-3.9	        GCC 7.3.1	    Bazel 3.7.2	    8.1	    11.2
# tensorflow-2.6.0	    3.6-3.9	        GCC 7.3.1	    Bazel 3.7.2	    8.1	    11.2
# tensorflow-2.5.0	    3.6-3.9	        GCC 7.3.1	    Bazel 3.7.2	    8.1	    11.2
# tensorflow-2.4.0	    3.6-3.8	        GCC 7.3.1	    Bazel 3.1.0	    8.0	    11.0
# tensorflow-2.3.0	    3.5-3.8	        GCC 7.3.1	    Bazel 3.1.0	    7.6	    10.1
# tensorflow-2.2.0	    3.5-3.8	        GCC 7.3.1	    Bazel 2.0.0	    7.6	    10.1
# tensorflow-2.1.0	    2.7, 3.5-3.7	GCC 7.3.1	    Bazel 0.27.1	7.6	    10.1
# tensorflow-2.0.0	    2.7, 3.3-3.7	GCC 7.3.1	    Bazel 0.26.1	7.4	    10.0

# Switch apt source to Taiwan's mirror
RUN sed -i 's@archive.ubuntu.com@tw.archive.ubuntu.com@g' /etc/apt/sources.list

# Timezone
RUN echo "tzdata tzdata/Areas select Asia" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Asia select Taipei" | debconf-set-selections && \
    apt-get update && apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/${USER}/.config/terminator/

WORKDIR /home/${USER}
USER ${USER}

RUN mkdir work
WORKDIR /home/${USER}/work

CMD ["terminator"]
USER $USER


