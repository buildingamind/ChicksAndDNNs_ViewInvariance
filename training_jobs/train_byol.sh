viewpoint=V11O2

for viewpoint in fork_0 ship_0
do
    python3 ../train_byol.py \
        --max_epochs 100 \
        --batch_size 512 \
        --data_dir /data/lpandey/Wood2024_LineDrawings_Dataset/Exp2/training/lines/${viewpoint} \
        --seed_val 0 \
        --dataset_size 80000 \
        --architecture 'resnet_2blocks' \
        --val_split 0.01 --exp_name paper_Lines/Exp2/byol10L/training_lines/${viewpoint}
done

        # --jitter_strength 0.5 \
        # --gaussian_blur \