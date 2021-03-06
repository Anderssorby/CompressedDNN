#!/bin/sh
#$ -S /bin/sh
# Script name
#$ -N train-wgan
# Job class name
#$ -jc gpu-container_g1_dev
#
# Docker container
# use one provided by list_containers
#$ -ac d=aip-gpinfo-03
#
# For interactive environment
# qrsh -jc gpu-container_g1_dev -ac d=aip-gpinfo-03

export PROJECT=CompressedDNN
if [[ ! -d ${PROJECT} ]]; then
 echo "Can't find $PROJECT."
 echo $HOME
else
 cd ${PROJECT}
fi

# Set env
. ./get_env.sh


# Parameters
MODEL=cifar10_wgan
EPOCH=10000
ACTION=train_model
GPU=0

python do.py --action ${ACTION} --model ${MODEL} --gpu ${GPU} --epoch ${EPOCH}
