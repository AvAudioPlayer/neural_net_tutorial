#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -M jungkyup@uci.edu
#$ -o gen-dis.out
#
#
#
#
module load sge
module load gcc/5.2.0
#
#
#
/auto/ugrad_space/jungkyup/anaconda2/bin/python generator-discriminator.py
