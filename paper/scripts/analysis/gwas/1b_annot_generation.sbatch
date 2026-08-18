#! /bin/bash
# author: Lisa Sikkema
# date: 2021.08.20

# submission of job to slurm cluster, to then generate an ldsc .annot file
# for a given celltype/cluster and chromosome.
#SBATCH -o sbatch_annot_%j.out
#SBATCH -e sbatch_annot_%j.out
#SBATCH -J LISA
#SBATCH -p cpu_p
#SBATCH --constraint="Lustre_File_System"
#SBATCH -c 1
#SBATCH --mem 6G
#SBATCH --time 0:40:00
#SBATCH --nice=10000
ct_name=$1
chr=$2
wd=$3
# adjust path as needed, it should be the path to your ldsc folder:
ldsc_path="$HOME/software/ldsc"
# to enable activation of conda environment within script
source $HOME/.bashrc
chmod 600 sbatch_annot_**.out
# activate ldsc conda environment
conda activate ldsc
cd $wd
python ${ldsc_path}/make_annot.py \
	--gene-set-file ../../../results/LDSC_GWAS_to_celltype/celltype_genesets/${ct_name}.GeneSet \
	--gene-coord-file ../../../results/LDSC_GWAS_to_celltype/ext_files/ENSG_coord.txt \
	--windowsize 100000 \
	--bimfile ../../../results/LDSC_GWAS_to_celltype/ext_files/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chr}.bim \
	--annot-file ../../../results/LDSC_GWAS_to_celltype/annots/${ct_name}.${chr}.annot.gz
