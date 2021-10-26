#!/bin/bash

NTS="100000"
TAG="MC-F0-NOM3"  #
BS="512"
TASK="MountainCarContinuous-v0" # 
NUMLAYERS="4"
BSDT="huber" #"L2" #"huber"
SAVEDIR="save"
CLR="0.001"
ALR="0.001"
ELR="0.001"
RPBC="50000"
TMT="deterministic"
EVALFREQ="5"
NOM="3" # (no effect if without --noisy_observation flag)
SPARSITY="-1.0" # NO EFFECT

mkdir -p ${SAVEDIR}

j=0
for i in {7..10}
do
    # Bisim-orig-IR
    j=$(((j+1) % 2))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig-IR_S${i} --bisim_dist $BSDT \
                             --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig-IR_$i.c.log &
    # Bisim-normed-IR
    j=$(((j+1) % 2))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_$i.c.log &
	# Bisim-normed-IR-ID
    j=$(((j+1) % 2))
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID_$i.c.log &
done

#
