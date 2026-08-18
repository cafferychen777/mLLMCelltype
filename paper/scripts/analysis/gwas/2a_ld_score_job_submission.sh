#! /bin/bash
# author: Lisa Sikkema
# date: 2021.06.08
# script to submit jobs for LD score calculation in a parallel fashion
# using a SLURM cluster

# get gene set file names, so that we can retrieve all celltypes/clusters of interest
file_names=`ls ../../../results/LDSC_GWAS_to_celltype/celltype_genesets`
suffix_to_remove=".GeneSet"
# store current directory as working directory:
wd=`pwd`
echo $wd
# loop through all file names (= cell types + control)
for fn in $file_names; do
	ct_name=${fn%"$suffix_to_remove"}
	echo "working on ${ct_name}"
	for chr in $(seq 1 22); do
		# # test conditional:
		# if [ $ct_name == "AT1" ] && [ $chr == 22 ]; then
		# condition on if file exists already or not (to complement earlier partly failed runs):
		outfile_prefix="../../../results/LDSC_GWAS_to_celltype/annots/${ct_name}.${chr}"
		if [ ! -f ${outfile_prefix}.l2.ldscore.gz ] || [ ! -f  ${outfile_prefix}.l2.M ] || [ ! -f  ${outfile_prefix}.l2.M_5_50 ]; then
			# submit job using sbatch script
			sbatch ./2b_ld_score_calculation.sbatch ${ct_name} ${chr} ${wd}
		fi
		# fi # only if test condition above is included
	done
done
