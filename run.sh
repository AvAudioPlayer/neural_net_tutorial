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
python generator-discriminator.py
