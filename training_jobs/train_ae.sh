viewpoint=V1O2

for viewpoint in setB_imp
do
    python3 ../train_ae.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/Wood2024_LineDrawings_Dataset/Exp3/training/lines/${viewpoint} \
        --seed_val 0 \
        --dataset_size 80000 \
        --enc_type resnet18_2b \
        --val_split 0.1 --exp_name paper_Wood2024_lineDrawings/Exp3/AE10L/80k/lines/${viewpoint}
done