# Towards Robust Bisimulation Metric Learning

Accepted to NeurIPS 2021.

## Usage 

Install the conda environment `dbc` in `conda_env.yml`.

Example: to run our `IR+ID` model on the modified ContinuousCartpole-v0 task (called "Noisy Sparse Cartpole" in our manuscript) with `N_m = 1`, run the following command:
    
    python train.py --domain_name $TASK --agent bisim --decoder_type identity --noisy_observation \
                    --encoder_type mlp --seed 0 --device 0 --noisy_dims 1 \
                    --replay_buffer_capacity 50000 --encoder_max_norm \
                    --intrinsic_reward_type forward_mean --latent_prior inverse_dynamics \
                    --sparsity_factor 0.01 --num_train_steps 50000 --batch_size 512

For additional usage, run `train.py --help`.

## Reproducing Results

Scripts for reproducing all experiments in the paper are in the subfolder `scripts`.

## Acknowledgments

Our code is based on the original Deep Bisimulation for Control code: [[paper](https://arxiv.org/abs/2006.10742)] [[code](https://github.com/facebookresearch/deep_bisim4control)], which is CC-BY-NC 4.0 licensed.

## Citing

If you use our codebase, please consider citing our paper:

    @inproceedings{NEURIPS2021_256bf8e6,
     author = {Kemertas, Mete and Aumentado-Armstrong, Tristan},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
     pages = {4764--4777},
     publisher = {Curran Associates, Inc.},
     title = {Towards Robust Bisimulation Metric Learning},
     url = {https://proceedings.neurips.cc/paper/2021/file/256bf8e6923a52fda8ddf7dc050a1148-Paper.pdf},
     volume = {34},
     year = {2021}
    }
