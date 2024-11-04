viewpoint=V1O2

for viewpoint in train1_80k
do
    python3 ../train_ae.py \
        --max_epochs 100 \
        --batch_size 128 \
        --data_dir /data/lpandey/UT_Austin_EgocentricDataset/output_64x64Squished/${viewpoint} \
        --seed_val 10 \
        --dataset_size 10000 \
        --enc_type resnet18 \
        --shuffle \
        --val_split 0.1 \
        --exp_name test_runs_nov3/${viewpoint}
done

# NOTES :
# --shuffle \
# --print_model \
