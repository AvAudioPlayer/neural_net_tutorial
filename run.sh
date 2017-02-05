#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -pe openmpi 64
#$ -o cifar10_smaller.out
#
#
#
#
module load sge
#
#
#
time /auto/ugrad_space/jungkyup/anaconda2/bin/python cifar10_smaller.py
