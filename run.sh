#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -pe openmpi 512
#$ -o cifar512.out
#
#
#
#
module load sge
#
#
#
time /auto/ugrad_space/jungkyup/anaconda2/bin/python cifar10_cnn.py
