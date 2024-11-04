viewpoint=V11O2

for viewpoint in ep0
do
    python3 ../train_vit_simclr.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/train3_80k/${viewpoint} \
        --seed_val 0 \
        --temporal \
        --temporal_mode '2+images' \
        --shuffle False \
        --window_size 3 \
        --head 1 \
        --drop_ep 0 \
        --val_split 0.05 --exp_name trial_runs_nov3/${viewpoint}
done
