#!/bin/bash

PROJECT_NAME=HSH-Taylor-Finetune-Hunyuan
DATA="$(date +"%m%d%H%M")"

SCRIPT_ROOT="$(cd "$(dirname "$0")"; pwd -P)"
cd $SCRIPT_ROOT/../..

# export VS_DEBUG=True

# export BOSS_CKPT_ENDPOINT=http://jssz-inner-boss.bilibili.co
# export BOSS_CKPT_ACCESS=sDrLN3WdwCf0sfmh
# export BOSS_CKPT_SECRET=3W3ItUsexS67C7PTHlitNeSkQguIwra2
# export BOSS_CKPT_BUCKET=virtualhuman
# export BOSS_CKPT_PATH=2D/tmp/weight
# export BOSS_CKPT_PATH=${BOSS_CKPT_PATH}/${PROJECT_NAME}/${DATA}

# echo "${BOSS_CKPT_PATH}"
# export HF_ENDPOINT=https://hf-mirror.com

# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0077.html
export HCCL_INTRA_ROCE_ENABLE=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0079.html
export HCCL_INTRA_PCIE_ENABLE=0
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0122.html
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0120.html
# export ASCEND_GLOBAL_LOG_LEVEL=1
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0091.html
# export HCCL_ENTRY_LOG_ENABLE=1
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0050.html
# export ASCEND_LAUNCH_BLOCKING=1
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0074.html
# export HCCL_CONNECT_TIMEOUT=300
# https://www.hiascend.com/document/detail/zh/canncommercial/80RC22/apiref/envvar/envref_07_0075.html
# export HCCL_EXEC_TIMEOUT=300
# https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/envref/envref_07_0053.html
# export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:2048,garbage_collection_threshold:0.6,expandable_segments:False
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline

# export WANDB_MODE=online
# export WANDB_API_KEY=40a8a90eeced7df8b5bf42c034c96d749f15bbd1

rm -rf kernel_meta

torchrun --nnodes 1 \
         --nproc_per_node 16 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=10000 \
         fastvideo/train.py \
         --seed 42 \
         --pretrained_model_name_or_path data/hunyuan \
         --dit_model_name_or_path data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
         --model_type "hunyuan" \
         --cache_dir data/.cache \
         --data_json_path data/Image-Vid-Finetune-HunYuan/videos2caption.json \
         --validation_prompt_dir data/Image-Vid-Finetune-HunYuan/validation \
         --gradient_checkpointing \
         --train_batch_size=1 \
         --num_latent_t 24 \
         --sp_size 8 \
         --train_sp_batch_size 1 \
         --dataloader_num_workers 4 \
         --gradient_accumulation_steps=1 \
         --max_train_steps=4000 \
         --learning_rate=5e-6 \
         --mixed_precision=bf16 \
         --checkpointing_steps=1000 \
         --validation_steps 100 \
         --validation_sampling_steps 16 \
         --checkpoints_total_limit 3 \
         --allow_tf32 \
         --ema_start_step 0 \
         --cfg 0.0 \
         --ema_decay 0.999 \
         --output_dir=data/outputs/${PROJECT_NAME} \
         --tracker_project_name ${PROJECT_NAME} \
         --validation_num_frames 93 \
         --validation_width 1280 \
         --validation_height 720 \
         --validation_guidance_scale "1.0" \
         --group_frame \
         --log_validation
