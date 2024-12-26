viewpoint=V11O2

for viewpoint in ep0
do
    python3 ../train_vit.py \
        --lars_wrapper \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/train3_80k/${viewpoint} \
        --seed_val 0 \
        --temporal \
        --window_size 3 \
        --image_size 64 \
        --patch_size 8 \
        --head 1 \
        --val_split 0.05 \
        --exp_name test_runs_dec26/${viewpoint}
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