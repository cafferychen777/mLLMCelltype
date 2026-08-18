#! /bin/bash
# author: Lisa Sikkema
# date: 2021.08.20
# Script to calculate LD scores (l2) between SNPs in "upregulated" gene regions
# for specific cell types, and all SNPs on the matching chromosome as included in
# hapmap3 snps. One LD score per hapmap SNP, calculating overall linkage between
# that SNP and all SNPs from .annot file (i.e. all SNPs close to upregulated
# genes for cell type).
# SBATCH part if for submission to a SLURM cluster, but can be skipped and run
# as a bash script.
#SBATCH -o sbatch_l2_%j.out
#SBATCH -e sbatch_l2_%j.out
#SBATCH -J LISA
#SBATCH -p cpu_p
#SBATCH -c 1
#SBATCH --mem 2G #2
#SBATCH --constraint="Lustre_File_System"
#SBATCH --time 6:30:00
#SBATCH --nice=10000
ct_name=$1
chr=$2
wd=$3
# adjust path as needed:
ldsc_path=$HOME/software/ldsc
# to enable conda environment activation from within script:
source $HOME/.bashrc
chmod 600 sbatch_l2_**.out
# activate conda environment
conda activate ldsc
# change to right directory:
cd $wd
echo "chr ${chr}"
python ${ldsc_path}/ldsc.py \
--l2 \
--bfile ../../../results/LDSC_GWAS_to_celltype/ext_files/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr} \
--ld-wind-cm 1 \
--annot ../../../results/LDSC_GWAS_to_celltype/annots/${ct_name}.${chr}.annot.gz \
--thin-annot \
--out ../../../results/LDSC_GWAS_to_celltype/annots/${ct_name}.${chr} \
--print-snps ../../../results/LDSC_GWAS_to_celltype/ext_files/hapmap3_snps/hm.${chr}.snp
# l2 flag specifies that we want to calculate l2 scores
# bfile: prefix of path to Plink file
# ld-wind-cm 1: use a 1 centiMorgan window for LD score calculation
# annot: location of matching annot file
# thin-annot: used because our annot files do not contain CHR, BP etc. information, but instead follow the exact order of our bim files
# out: location of out file
# print-snps: specifies how to print (only SNPs that are listed), trick to save memory, should not affect output
