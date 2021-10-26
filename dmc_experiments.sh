#!/bin/bash

TAG="cartpole-swingup-sparse"

for i in {1..5}
do
    # SAC
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cartpole --task_name swingup_sparse \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent baseline \
                        --num_procs 16 --replay_buffer_capacity 160000 --work_dir ./${TAG}_SAC_S${i}  &> "$TAG"_SAC_$i.c.log
done

for i in {1..5}
do
    # Bisim
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cartpole --task_name swingup_sparse \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent bisim \
                        --num_procs 16 --replay_buffer_capacity 160000 --work_dir ./${TAG}_DBC_S${i} &> "$TAG"_DBC_$i.c.log
done

for i in {1..5}
do
    # Bisim-ours
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cartpole --task_name swingup_sparse \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent bisim \
                        --intrinsic_reward_type forward_mean \
                        --num_procs 16 --replay_buffer_capacity 160000 \
                        --encoder_max_norm --latent_prior inverse_dynamics \
                        --intrinsic_reward_weight 1 --intrinsic_reward_max 0.1 --latent_prior_weight 10.  --work_dir ./${TAG}_OURS_S${i} &> "$TAG"_DBC-normed-IR-ID_$i.c.log
done

TAG="cheetah-run"

for i in {1..5}
do
    # SAC
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cheetah --task_name run \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent baseline \
                        --num_procs 16 --replay_buffer_capacity 160000 --work_dir ./${TAG}_SAC_S${i}  &> "$TAG"_SAC_$i.c.log
done

for i in {1..5}
do
    # Bisim
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cheetah --task_name run \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent bisim \
                        --num_procs 16 --replay_buffer_capacity 160000 --work_dir ./${TAG}_DBC_S${i} &> "$TAG"_DBC_$i.c.log
done

for i in {1..5}
do
    # Bisim-ours
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL="egl" python -u train_dmc_parallel.py --domain_name cheetah --task_name run \
                        --encoder_type pixel --decoder_type identity \
                        --action_repeat 4 --seed $i --agent bisim \
                        --intrinsic_reward_type forward_mean \
                        --num_procs 16 --replay_buffer_capacity 160000 \
                        --encoder_max_norm --latent_prior inverse_dynamics \
                        --intrinsic_reward_weight 1 --intrinsic_reward_max 0.1 --latent_prior_weight 10.  --work_dir ./${TAG}_OURS_S${i} &> "$TAG"_DBC-normed-IR-ID_$i.c.log
done