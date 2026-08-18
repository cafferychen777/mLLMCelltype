#! /bin/bash
# author: Lisa Sikkema
# date: 2021.06.08
# script to submit jobs for LD score regression in a parallel fashion
# using a SLURM cluster

for dataset_name in Sakornsakolpat_COPD McKay_lung_cancer Howard_depression Han_asthma Allen_IPF; do # plus Shrine_lung_function
	sbatch 4b_ldsc_regression.sh $dataset_name
done
