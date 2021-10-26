#!/bin/bash

NTS="100000"
TAG="MC-F2-NOM2"  #"MC-mubd_norm"
BS="512"
TASK="MountainCarContinuous-v0" # ContinuousCartpole-v0
ND="4" # num_devices
NUMLAYERS="4"
BSDT="huber" #"L2" #"huber"
SAVEDIR="save"
CLR="0.001"
ALR="0.001"
ELR="0.001"
RPBC="50000"
TMT="deterministic"
EVALFREQ="5"
NOM="2" # (no effect if without --noisy_observation flag)
SPARSITY="-1.0" # NO EFFECT

mkdir -p ${SAVEDIR}

for i in {5..9}
do
    ## SAC
    #nohup python -u train.py --domain_name $TASK --agent baseline --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --work_dir ${SAVEDIR}/${TAG}_SAC_S${i} --bisim_dist $BSDT \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_SAC_$i.c.log &
    ## Bisim
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-orig_S${i} --bisim_dist $BSDT \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig_$i.c.log &
    ## Bisim-normed
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed_S${i} --bisim_dist $BSDT --encoder_max_norm \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed_$i.c.log &
    # Bisim-normed-ID
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-ID_$i.c.log &
    # Bisim-orig-IR
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig-IR_S${i} --bisim_dist $BSDT \
                             --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig-IR_$i.c.log &
    # Bisim-normed-IR
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_$i.c.log &
	# Bisim-normed-IR-ID
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --num_eval_episodes 10 --eval_freq $EVALFREQ \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID_$i.c.log &
	## Bisim-normed-IR-ID-1
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID5_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} --latent_prior_weight 5.0 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID5_$i.c.log &
	## Bisim-normed-IR-ID-2
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID10_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT}  --latent_prior_weight 10.0 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID10_$i.c.log &
	## Bisim-normed-IR-ID-3
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID20_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} --latent_prior_weight 20.0 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID20_$i.c.log &
    # Bisim-normed-IR-LLR
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity  \
    #                         --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr 0.001 --actor_lr 0.001 --encoder_lr 0.0001 \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_LLR_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_LLR_$i.c.log &
	# Bisim-normed-IR-ID-LLR
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity  \
    #                         --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr 0.001 --actor_lr 0.001 --encoder_lr 0.0001 \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_LLR_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID_LLR_$i.c.log &
    # Bisim-alt
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity  \
    #                         --encoder_type mlp --seed $i --device 0 --c_R 0.5 --c_T 0.5 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-alt_S${i} --bisim_dist $BSDT \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-alt_$i.c.log &
	
    # Bisim-normed-IR+mubd_norm
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --norm_ir_by_mu_bd \
    #                         --encoder_type mlp --seed $i --device 0 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IRmbn_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IRmbn_$i.c.log &
	# Bisim-normed-IR-ID_mubd_norm
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --norm_ir_by_mu_bd \
    #                         --encoder_type mlp --seed $i --device 1 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --num_eval_episodes 10 --eval_freq $EVALFREQ \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IRmbn_ID_S${i} --bisim_dist $BSDT \
    #                         --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IRmbn_ID_$i.c.log &
    ####
    
    #CD=$(($i % $ND))
   	#nohup python -u train.py --domain_name $TASK --agent baseline --decoder_type identity \
    #                         --replay_buffer_capacity $RPBS \
    #                         --encoder_type mlp --seed $i --device 0 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS >> "$TAG"_SAC_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --replay_buffer_capacity $RPBS \
    #                         --encoder_type mlp --seed $i --device 1 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --replay_buffer_capacity $RPBS \
    #                         --encoder_type mlp --seed $i --device 2 --c_R 0.5 --c_T 0.5 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-alt_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --encoder_type mlp --seed $i --device 0 --c_R 0.5 --c_T 0.5 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS \
    #                         --replay_buffer_capacity $RPBS \
    #                         --intrinsic_reward_type forward_mean >> "$TAG"_IR-alt_$i.c.log & 
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --encoder_type mlp --seed $i --device 1 --c_R 0.5 --c_T 0.5 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS \
    #                         --replay_buffer_capacity $RPBS \
    #                         --intrinsic_reward_type forward_mean \
    #                         --latent_prior inverse_dynamics >> "$TAG"_IR_ID-alt_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --encoder_type mlp --seed $i --device 2 --c_R 0.5 --c_T 0.5 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS \
    #                         --replay_buffer_capacity $RPBS \
    #                         --intrinsic_reward_type forward_mean \
    #                         --latent_prior TK+ID >> "$TAG"_IR_ID_TK-alt_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity \
    #                         --encoder_type mlp --seed $i --device 1 --c_R 0.5 --c_T 0.5 \
    #                         --sparsity_factor 0.01 --num_train_steps $NTS --batch_size $BS \
    #                         --replay_buffer_capacity $RPBS \
    #                         --latent_prior inverse_dynamics >> "$TAG"_ID-alt_$i.c.log &
done

#
