#!/bin/bash
# To submit SLURM job for label transfer per dataset
# specify paths
log_dir="."
output_dir="../results/HLCA_ext_label_transfer/per_dataset"
# original number of cores: 24, mem: 200G conda run -n nn
# Run label transfer python script for every dataset. The python script will use the stored embedding of the HLCA core and extension combined, plus the annotations in the core, to perform the label transfer.
for i in Meyer_2021_5prime # re-running only meyer data now, otherwise use: Banovich_Kropski_2020 Barbry_unpubl Budinger_2020 Duong_lungMAP_unpubl Eils_2020 Gomperts2021_UCLA Gomperts_2021_CFF Gomperts_2021_CSMC Janssen_2020 Kaminski_2020 Lafyatis_2019 Lambrechts_2021 MeyerNikolic_unpubl_UCL Meyer_2021_3prime Meyer_2021_5prime Misharin_Budinger_2018 Peer_Massague_2020 Regev_2021_Cryo Regev_2021_Fresh Regev_2021_Nuclei Schiller_2020 Schiller_2021 Schultze_unpubl Shalek_2018 Sheppard_2020 Sims_2019 Sun_2020_batch1 Sun_2020_batch2 Sun_2020_batch3 Sun_2020_batch4 Tata_unpubl Thienpont_2018_10Xv1 Thienpont_2018_10Xv2 Wunderink_2021_cryo Wunderink_2021_fresh Xu_2020_LAM1_3 Zhang_2021
do
srun -o ${log_dir}/label_transfer_$i.o.log -e ${log_dir}/label_transfer_$i.e.log -p interactive_cpu_p --constraint="Lustre_File_System" -c 4 -t 7:00:00 --mem=40G --export=ALL conda run -n HLCA_basic python ./label_transfer.py $i $output_dir &
done
echo "finished knn classifier for every dataset"
