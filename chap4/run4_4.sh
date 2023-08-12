python train.py \
    --path_train_csv ../data/gldv2_micro/train.csv \
    --path_val_csv ../data/gldv2_micro/val.csv \
    --outputs_dir ../outputs \
    --gldv2_micro_path ../data/gldv2_micro \
    --batch_size 8 \
    --device cpu
python evaluate.py \
    --roxford5k_root_dir ../data \
    --outputs_dir ../outputs \
    --device cpu