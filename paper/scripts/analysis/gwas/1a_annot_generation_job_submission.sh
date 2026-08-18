#! /bin/bash
# author: Lisa Sikkema
# date: 2021.08.20
# script to submit jobs for the generation of annot files for LDSC analysis in a parallel fashion,
# using a SLURM cluster

# get gene set file names, so that we can retrieve all celltypes/clusters of interest
file_names=`ls ../../../results/LDSC_GWAS_to_celltype/celltype_genesets`
suffix_to_remove=".GeneSet"
# store current directory as working directory:
wd=`pwd`
# loop through all file names (= cell types + control)
for fn in $file_names; do
	ct_name=${fn%"$suffix_to_remove"}
	echo "working on ${ct_name}"
	for chr in $(seq 1 22); do
		outfile_name_expected="../../../results/LDSC_GWAS_to_celltype/annots/${ct_name}.${chr}.annot.gz"
		if [ ! -f ${outfile_name_expected} ]; then # only go through iteration if file doesn't exist yet
		# # test conditional:
		# if [ $ct_name == "AT1" ] && [ $chr == 22 ]; then
		# submit job using sbatch script
			sbatch ./1b_annot_generation.sbatch ${ct_name} ${chr} ${wd}
		# fi
		fi
	done
done
