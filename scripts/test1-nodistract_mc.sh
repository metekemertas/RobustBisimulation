#!/bin/bash

NTS="100000"
TAG="MC-F1-NO_NOM"  
BS="512"
TASK="MountainCarContinuous-v0" #
ND="4" # num_devices
NUMLAYERS="4"
BSDT="huber" #"L2" #"huber"
SAVEDIR="save"
CLR="0.001"
ALR="0.001"
ELR="0.001"
RPBC="50000"
TMT="deterministic"
EVALFREQ="10"
NOM="1" # NO EFFECT (without --noisy_observation flag)
SPARSITY="-1.0" # NO EFFECT

mkdir -p ${SAVEDIR}

for i in {8..10}
do
    # SAC
    nohup python -u train.py --domain_name $TASK --agent baseline --decoder_type identity \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_SAC_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_SAC_$i.c.log &
    # Bisim
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
                             --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig_$i.c.log &
    # Bisim-normed
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed_S${i} --bisim_dist $BSDT --encoder_max_norm \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed_$i.c.log &
    # Bisim-normed-ID
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
                             --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-ID_$i.c.log &
    # Bisim-orig-IR
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig-IR_S${i} --bisim_dist $BSDT \
                             --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig-IR_$i.c.log &
    # Bisim-normed-IR
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity  \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_$i.c.log &
	# Bisim-normed-IR-ID
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity  \
                             --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID_$i.c.log &
done

#
