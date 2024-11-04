viewpoint=V1O1

for viewpoint in ep0
do
    python3 ../train_simclr.py \
        --lars_wrapper \
        --max_epochs 100 \
        --batch_size 512 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/train1_80k/${viewpoint} \
        --seed_val 0 \
        --architecture resnet_2blocks \
        --drop_ep 0 \
        --temporal \
        --window_size 3 \
        --temporal_mode '2+images' \
        --val_split 0.05 \
        --exp_name test_runs/${viewpoint}
done