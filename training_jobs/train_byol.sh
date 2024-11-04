viewpoint=V11O2

for viewpoint in train1_80k
do
    python3 ../train_byol.py \
        --max_epochs 100 \
        --batch_size 512 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/${viewpoint} \
        --seed_val 0 \
        --dataset_size 80000 \
        --backbone resnet18_2blocks \
        --val_split 0.01 \
        --shuffle \
        --aug False \
        --print_model \
        --exp_name test_runs_nov3/${viewpoint}
done

# NOTES :
# --shuffle \
# --print_model \