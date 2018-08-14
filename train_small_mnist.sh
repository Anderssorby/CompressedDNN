#!/bin/sh
#$ -S /bin/sh
# Script name
#$ -N train_small_mnist
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
if [ ! -d ${PROJECT} ]; then
 echo "Can't find $PROJECT is ill defined."
else
 cd ${PROJECT}
fi

. ./get_env.sh

# Parameters
MODEL=mnist_vgg2
EPOCH=1000
ACTION=train_model
GPU=0
PREFIX=small500
UNIT=500

python do.py --action ${ACTION} --model ${MODEL} --prefix ${PREFIX} --gpu ${GPU} --epoch ${EPOCH} --unit ${UNIT}
python do.py --action calc_eigs --model ${MODEL} --prefix ${PREFIX} --gpu ${GPU}
python do.py --action calc_dof --model ${MODEL} --prefix ${PREFIX} --gpu ${GPU}
python do.py --action update_architecture --model ${MODEL} --prefix ${PREFIX}
