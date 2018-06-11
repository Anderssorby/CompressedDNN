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
. /fefs/opt/dgx/env_set/aip-gpinfo.sh
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

python train.py --model wgan --gpu 0 --epoch 20
