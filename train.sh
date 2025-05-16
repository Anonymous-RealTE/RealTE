CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=8888 train.py \
    --train_batch_size 2\
    --ratio 0.5 \
    --resume ./exp/final_ckpt/  \
    --mixed_precision "fp16" \
    --enable_xformers_memory_efficient_attention \
    --output_dir ./exp/20250513_ckpt \
    --data_root LineData \
    --datasets "IC13_857/train-50k-1" 
    