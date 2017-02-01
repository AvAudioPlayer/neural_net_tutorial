#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -pe openmpi 128
#$ -o mnist128.out
#
#
#
#
module load sge
#
#
#
time /auto/ugrad_space/jungkyup/anaconda2/bin/python mnist.py
