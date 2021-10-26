#!/bin/bash

NTS="50000"
TAG="CP-nom2-s0d01"
BS="512"
#TASK="Pendulum-v0" # "MountainCarContinuous-v0" # ContinuousCartpole-v0
TASK="ContinuousCartpole-v0"
SPARSITY="0.01" # Controls contraction of reward area
SAVEDIR="save"
NOM="2" # "3" = working, as desired; "4" = requires IR
CLR="0.001"
ALR="0.001"
ELR="0.001"
NUMLAYERS="4"
BSDT="huber"  # "L2" # "huber"
RPBC="50000"
TMT="deterministic"

NDEVICES=4

mkdir -p ${SAVEDIR}

for i in {1..10}
do
    echo "Seed $i"

    # SAC
    j=$(((j+1) % NDEVICES))
    echo "SAC on $j"
    nohup python -u train.py --domain_name $TASK --agent baseline --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_SAC_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_SAC_$i.c.log &
    # Bisim
    j=$(((j+1) % NDEVICES))
    echo "DBC on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig_$i.c.log &
    # Bisim-IR
    j=$(((j+1) % NDEVICES))
    echo "DBC-IR on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig-IR_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --intrinsic_reward_type forward_mean \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig-IR_$i.c.log &
    # Bisim-IR-ID
    j=$(((j+1) % NDEVICES))
    echo "DBC-IR-ID on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-orig-IR-ID_S${i} --bisim_dist $BSDT \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-orig-IR-ID_$i.c.log &
    # Bisim-normed
    j=$(((j+1) % NDEVICES))
    echo "DBC-normed on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed_S${i} --bisim_dist $BSDT --encoder_max_norm \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed_$i.c.log &
    # Bisim-normed-IR
    j=$(((j+1) % NDEVICES))
    echo "DBC-normed-IR on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_$i.c.log &
    # Bisim-normed-IR-ID
    j=$(((j+1) % NDEVICES))
    echo "DBC-normed-IR-ID on $j"
    nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                             --encoder_type mlp --seed $i --device $j --num_layers $NUMLAYERS \
                             --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
                             --work_dir ${SAVEDIR}/${TAG}_DBC-normed-IR_ID_S${i} --bisim_dist $BSDT \
                             --encoder_max_norm --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                             --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
                             --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-normed-IR_ID_$i.c.log &
    # Bisim-alt
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device $i --c_R 0.5 --c_T 0.5 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --work_dir ${SAVEDIR}/${TAG}_DBC-alt_S${i} --bisim_dist $BSDT \
    #                         --replay_buffer_capacity ${RPBC} --transition_model_type ${TMT} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS >> "$TAG"_DBC-alt_$i.c.log &
    ## Bisim-alt-IR
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 3 --c_R 0.5 --c_T 0.5 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --work_dir ${SAVEDIR}/${TASK}_IR-alt_S${i} --bisim_dist $BSDT \
    #                         --intrinsic_reward_type forward_mean >> "$TAG"_IR-alt_$i.c.log &
    ## Bisim-alt-IR-ID
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 1 --c_R 0.5 --c_T 0.5 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --intrinsic_reward_type forward_mean \
    #                         --work_dir ${SAVEDIR}/${TASK}_IR_ID-alt_S${i} --bisim_dist $BSDT \
    #                         --latent_prior inverse_dynamics >> "$TAG"_IR_ID-alt_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 2 --c_R 0.5 --c_T 0.5 --num_layers $NUMLAYERS \
    #                         --noisy_dims ${NOM} --critic_lr ${CLR} --actor_lr ${ALR} --encoder_lr ${ELR} \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --work_dir ${SAVEDIR}/${TASK}_ID-alt_S${i} --bisim_dist $BSDT \
    #                         --latent_prior inverse_dynamics >> "$TAG"_ID-alt_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 0 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --intrinsic_reward_type forward_mean >> "$TAG"_IR-orig_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 1 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --intrinsic_reward_type forward_mean \
    #                         --latent_prior inverse_dynamics >> "$TAG"_IR_ID-orig_$i.c.log &
    #nohup python -u train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
    #                         --encoder_type mlp --seed $i --device 2 \
    #                         --sparsity_factor $SPARSITY --num_train_steps $NTS --batch_size $BS \
    #                         --latent_prior inverse_dynamics >> "$TAG"_ID-orig_$i.c.log &
done


#
