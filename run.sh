#!/bin/bash
#$ -N train
#$ -cwd
#$ -P one_slot
#$ -pe smp 20
#$ -l h_rt=02:00:00
#$ -o out.$JOB_ID.txt
#$ -e err.$JOB_ID.txt

python src/train.py