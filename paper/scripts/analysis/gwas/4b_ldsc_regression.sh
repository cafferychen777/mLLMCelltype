#! /bin/bash
# author: Lisa Sikkema
# date: 2022.07.22
# sbatch script to run ldsc regression in a slurm job
#SBATCH -o sbatch_ldsc_regr_%j.out
#SBATCH -e sbatch_ldsc_regr_%j.out
#SBATCH -J ldsc_regression
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem 4G #
#SBATCH --constraint="Lustre_File_System"
#SBATCH --time 6:30:00
#SBATCH --nice=10000
# choose GWAS results of interest:
# choose from: "Shrine_lung_function", "Sakornsakolpat_COPD", "McKay_lung_cancer", "Howard_depression", "Han_asthma", "Allen_IPF"
GWAS_study=$1 #"Han_asthma"
# set matching dataset name, which is often the same but in some cases not:
dataset_name=${GWAS_study}
if [ ${GWAS_study} = "Shrine_lung_function" ]; then dataset_name="Shrine_lung_function_fvc"; fi
if [ ${GWAS_study} = "McKay_lung_cancer" ]; then dataset_name="McKay_lung_cancer_adenocarcinoma"; fi
# print for logging:
echo "Working on GWAS study $GWAS_study, with dataset name $dataset_name"
# adjust as needed:
ldsc_path=$HOME/software/ldsc
# activate ldsc environment
# to enable conda environment activation from within script:
source $HOME/.bashrc
# adapt permissions for out script:
chmod 600 sbatch_ldsc_regr_**.out
# now activate conda environment
conda activate ldsc
# and run python script:
${ldsc_path}/ldsc.py \
--h2-cts ../../../data_GWAS_non_public/$GWAS_study/${dataset_name}.sumstats.gz \
--ref-ld-chr ../../../results/LDSC_GWAS_to_celltype/ext_files/1000G_EUR_Phase3_baseline/baseline. \
--out ../../../results/LDSC_GWAS_to_celltype/ldsc_results/${dataset_name}_1000genes \
--ref-ld-chr-cts ../../../results/LDSC_GWAS_to_celltype/cts.ldcts \
--w-ld-chr ../../../results/LDSC_GWAS_to_celltype/ext_files/weights_hm3_no_hla/weights. \
# h2-cts flag specifies that we want to do a celltype specific analysis. Put the sumstats file path here
# ref-ld-chr: location of baseline model files (chr part to specify that files are split per chromosome)
# ref-ld-chr-cts Path to file that specifies location of cell type an control annotation files
# w-ld-chr location of LD scores used for regression weights. Using hapmap3 file is recommended
