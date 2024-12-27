viewpoint=V1O1

for viewpoint in ep0
do
    python3 ../train_simclr.py \
        --lars_wrapper \
        --max_epochs 100 \
        --batch_size 512 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/train1_80k/${viewpoint} \
        --seed_val 0 \
        --backbone resnet_2blocks \
        --temporal \
        --window_size 3 \
        --val_split 0.05 \
        --aug False \
        --loss_ver v0 \
        --exp_name test_runs_dec27/${viewpoint}
done

# NOTES :
# --shuffle \
# --print_model \
# set window_size in the range [1,4]
# choose loss function version from v0 and v1

#  --shuffle_frames \
#  --loss_ver v0 \
#  --shuffle_temporalWindows \
#  --dataset_size 10000 \
#  --dataloader_shuffle \
