NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
WANDB_API_KEY=4bc3aa1826820be83c7e7ee086cf7256f7d40924 \
IMAGE_TRAIN_SIZE=252 \
USE_IMG_EMBED_AS_INPUT=false \
TOKEN_LOSS_WIGHT=0.1 \
MAX_PIXELS=262640 \
nproc_per_node=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model '/mnt/nas1/zhanghong/project/all2all/Qwen2__5-VL-7B-Instruct-All2All' \
    --model_type 'qwen2_5_all2all' \
    --template 'qwen2_5_all2all' \
    --train_type full \
    --dataset '/mnt/nas1/zhanghong/project/all2all/ms-swift/data/all2all/stage2/all2all_stage2_msg.jsonl' \
    --resume_from_checkpoint '/mnt/nas1/zhanghong/project/all2all/ms-swift/output/v83-20250401-140850/checkpoint-35088' \
    --resume_only_model true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --freeze_vit true \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --eval_steps 3000 \
    --save_steps 3000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --max_length 4096 \
    --deepspeed zero3 \
    --report_to wandb \


# data/all2all/merged_data/all2all_swift_1.6M.jsonl
# /mnt/nas1/zhanghong/project/all2all/data_preprocess/data/all2all/merged_data/all2all_eligen_swift.jsonl
# data/all2all/merged_data/all2all_swift_test.jsonl
