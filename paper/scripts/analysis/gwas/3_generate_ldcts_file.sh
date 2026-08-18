#! /bin/bash
# Author: Lisa Sikkema
# Date: 2021.06.08
# script to automatically generate the ldcts file needed to run LDSC celltype regression
# (for ref-ld-chr-cts flag in ldsc.py)

# store path to annot files, !! relative to the folder from which the ldsc regression is run!!!
# in our case, that is here:
ann_files_path_prefix="../../../results/LDSC_GWAS_to_celltype/annots"
# set file path to write to:
ldsc_file_name="../../../results/LDSC_GWAS_to_celltype/cts.ldcts"
# get gene set file names, so that we can retrieve all celltypes/clusters of interest
file_names=`ls ../../../results/LDSC_GWAS_to_celltype/celltype_genesets`
suffix_to_remove=".GeneSet"
echo "Make sure the file ${ldsc_file_name} does not already exist, we will only append lines!"
# loop through all file names (= cell types + control)
for fn in $file_names; do
	ct_name=${fn%"$suffix_to_remove"}
	if [ ${ct_name} != "control" ]; then
		printf "${ct_name}	${ann_files_path_prefix}/${ct_name}.,${ann_files_path_prefix}/control.\n" >> ${ldsc_file_name}
	fi
done
