#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -pe openmpi 16
#$ -o gen-dis.out
#
#
#
#
module load sge
#
#
#
/auto/ugrad_space/jungkyup/anaconda2/bin/python generator-discriminator.py
