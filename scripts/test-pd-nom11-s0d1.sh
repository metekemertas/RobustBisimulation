#!/bin/bash

NTS="100000"
TAG="PD-F1-nom11-s0d1"
BS="512"
TASK="SparsePendulum-v0"
SPARSITY="0.1" # Controls angular extent of reward for pendulum (degrees)
NUMLAYERS="4"
NOM="11"
BSDT="huber" #"L2" 
SAVEDIR="save"
CLR="0.001"
ALR="0.001"
ELR="0.001"
RPBC="50000"
TMT="deterministic"

mkdir -p ${SAVEDIR}

for i in {1..10}
do
    # SAC
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent baseline --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_SAC_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_SAC_$i.c.log &
    # Bisim
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_DBC-orig_$i.c.log &
    # Bisim-normed-NL2
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers 2 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-NL2_S${i} --bisim_dist $BSDT --encoder_max_norm \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_DBC-normed-NL2_$i.c.log &
    # Bisim-normed-IR-NL2
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers 2 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_NL2_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_DBC-normed-IR_NL2_$i.c.log &
    # Bisim-normed-ID-NL2
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers 2 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-ID_NL2_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_DBC-normed-ID_NL2_$i.c.log &
    # Bisim-normed-IR-ID-NL2
    j=$(((j+1) % 4))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers 2 \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq 5 \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_NL2_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS &> "$TAG"_DBC-normed-IR_ID_NL2_$i.c.log &
done

#
