#!/bin/sh
#$ -S /bin/sh
# Script name
#$ -N train
# Job class name
#$ -jc gpu-container_g1_dev
#
# Docker container
# use one provided by list_containers
#$ -ac d=aip-gpinfo-03
#
# For interactive environment
# qrsh -jc gpu-container_g1_dev -ac d=aip-gpinfo-03


# Set env
if [ -f /fefs/opt/dgx/env_set/aip-gpinfo.sh ]; then
 source /fefs/opt/dgx/env_set/aip-gpinfo.sh
else
 echo "Assuming test"
fi

# print date and time
echo Time is `date`
echo Directory is `pwd`
python --version

export PROJECT=CompressedDNN
if [ ! -d ${PROJECT} ]; then
 echo "Project $PROJECT is ill defined."
 echo $HOME
 exit
else
 cd ${PROJECT}
fi

# Parameters
MODEL=cifar10_wgan
EPOCH=10000
ACTION=train_model
GPU=0

python do.py --action ${ACTION} --model ${MODEL} --gpu ${GPU} --epoch ${EPOCH}
