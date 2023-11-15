#!/usr/bin/env bash

file_dir="$(dirname $0)"

# Get WorkSpace parameters
workspace=$(readlink -f "${0}")
workspace=${workspace%_ws*}
workspace_name=${workspace##*/}
workspace_path=${workspace}"_ws"

# Docker image and container name
image_name=$(echo ${workspace_name} | tr '[:upper:]' '[:lower:]')
container_name=${workspace_name}


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CONFIG_PATH="$SCRIPT_DIR/config/shell/terminator/config"

xhost +


docker run --runtime nvidia \
    -it --rm --network=host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -e XAUTHORITY=${HOME}/.Xauthority \
    -v ${HOME}/.Xauthority:/home/${user}/.Xauthority \
    -v /run/user/$(id -u)/:/run/user/$(id -u) \
    -v ./config/shell/terminator/config:/home/${USER}/.config/terminator/config \
    -v ${workspace_path}:${HOME}/work \
    --name $container_name $(id -un)/${image_name}