FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

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
ARG GID=${UID}
ARG SHELL=/bin/bash

# 創建一個名為initial的用戶和用戶組
# RUN groupadd -g $GID $GROUP && \
#     useradd -m -u $UID -g $GID -s /bin/bash $USER

## Setup users and groups
RUN groupadd --gid ${GID} ${GROUP} \
  && useradd --gid ${GID} --uid ${UID} -ms ${SHELL} ${USER} \
  && mkdir -p /etc/sudoers.d \
  && echo "${USER}:x:${UID}:${UID}:${USER},,,:$HOME:${shell}" >> /etc/passwd \
  && echo "${USER}:x:${UID}:" >> /etc/group \
  && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" \
  && chmod 0440 "/etc/sudoers.d/${USER}"

# Update and Install Packages
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y at-spi2-core
RUN apt-get install -y htop git wget curl
RUN apt-get install -y byobu zsh
RUN apt-get install -y terminator 
RUN apt-get install -y dbus-x11 libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 libxtst6 libinih-dev
RUN apt-get install -y nano vim
RUN apt-get install -y gnome-terminal libcanberra-gtk-module libcanberra-gtk3-module
RUN apt-get install -y python3-wheel python3-pip python3-dev python3-setuptools
# RUN apt-get install -y libopencv-dev
RUN apt-get install -y openssh-server
RUN apt-get install -y cmake
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN apt update
RUN apt install -y python3-tk

# 安裝opencv與tensorflow
# RUN pip3 install opencv-python tensorflow==2.11.0 matplotlib

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


