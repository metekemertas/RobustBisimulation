#!/bin/bash

# Use to start the simulator on device 1 with the specified port open
# CUDA_VISIBLE_DEVICES=1 bash CarlaUE4.sh -fps 20 -carla-world-port=2010

DOMAIN=carla098
TASK=highway
SEED=9
SAVEDIR=./save
AGENT=bisim
ESTRIDE=1 # Originally 1 in 0.9.8, 2 in 0.9.6
TRANSITION_MODEL=probabilistic #ensemble # probabilistic # originally not specified
DEVICE=1

mkdir -p ${SAVEDIR}

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --agent ${AGENT} \
    --device ${DEVICE} \
    --init_steps 1000 \
    --num_train_steps 100000 \
    --encoder_type pixelCarla098 \
    --decoder_type pixel \
    --img_source video \
    --resource_files 'distractors/*.mp4' \
    --action_repeat 4 \
    --critic_tau 0.01 \
    --encoder_tau 0.05 \
    --encoder_stride ${ESTRIDE} \
    --decoder_weight_lambda 0.0000001 \
    --hidden_dim 1024 \
    --total_frames 10000 \
    --num_filters 32 \
    --batch_size 128 \
    --init_temperature 0.1 \
    --alpha_lr 1e-4 \
    --alpha_beta 0.5 \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${AGENT}_S${SEED}-mod \
    --seed ${SEED} \
    --transition_model_type ${TRANSITION_MODEL} \
    --frame_stack 3 \
    --image_size 84 \
    --eval_freq 100 \
    --num_eval_episodes 25 \
    --replay_buffer_capacity 200000 \
    --c_R 0.5 \
    --c_T 0.5 \
    --intrinsic_reward_type forward_mean \
    --intrinsic_reward_weight 1.0 \
    --intrinsic_reward_max 1.0 \
    --latent_prior inverse_dynamics \
    --latent_prior_weight 1.0 
    #--port 2010
    #--render

# Eval_freq is by number of episodes
# Replay buffer originally 10^6
# num_train_steps (according to paper) of 100K

