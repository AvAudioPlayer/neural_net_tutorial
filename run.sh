#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -pe openmpi 32
#$ -o cifar32omp.out
#
#
#
#
module load sge
#
#
#
time /auto/ugrad_space/jungkyup/anaconda2/bin/python cifar10_cnn.py
