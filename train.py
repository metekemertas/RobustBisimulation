# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys, math

LOAD_CARLA = False

import numpy as np
import torch
import argparse
import gym
import time
import json
import dmc2gym

import utils
from logger import Logger
from video import VideoRecorder

from agent.baseline_agent import BaselineAgent
from agent.bisim_agent import BisimAgent
from agent.deepmdp_agent import DeepMDPAgent
from gym_utils import wrappers

if LOAD_CARLA:
    from agents.navigation.carla_env import CarlaEnv

def parse_args():

    allowed_tasks = [ 'MountainCarContinuous-v0',
                      'Pendulum-v0',
                      'SparsePendulum-v0',
                      'ContinuousCartpole-v0' ]

    parser = argparse.ArgumentParser()
    # environment
    #parser.add_argument('--domain_name', default='cheetah') # 
    parser.add_argument('--domain_name', 
                        default = 'ContinuousCartpole-v0',
                        help = 'Choose from' + str(allowed_tasks)) #
    parser.add_argument('--task_name', default='run')
    ##
    parser.add_argument('--sparsity_factor', default=1., type=float)
    ##
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # noisy observation
    parser.add_argument('--noisy_observation', action='store_true')
    parser.add_argument('--noisy_dims', default=5, type=int,
                        help = 'Multiplier for adding noise dims for distraction')
    parser.add_argument('--noise_std', default=1., type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    # eval
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='mlp', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity', 'mlp'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--bisim_dist', default='huber', type=str, choices=['L1', 'L2', 'huber'])
    parser.add_argument('--encoder_max_norm', default=False, action='store_true')
    parser.add_argument('--c_R', default=1., type=float) # 
    parser.add_argument('--c_T', default=None, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, 
                        choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--num_updates', default=1, type=int,
                        help = 'Number of agent updates per env step')
    # intrinsic rewards
    parser.add_argument('--intrinsic_reward_type', type=str, choices=['none', 'forward_mean', 'forward_dist'],
                        default = 'none', help = 'Type of intrinsic reward to use')
    parser.add_argument('--intrinsic_reward_weight', type=float, #default=2.0,
                        help = 'Weight on intrinsic reward (scale/eta)')
    parser.add_argument('--intrinsic_reward_max', type=float, #default=5.0,
                        help = 'Maximum allowed intrinsic reward value (before weighting)')
    parser.add_argument('--latent_prior', type=str, default = 'none',
                        choices=[ 'none', 'inverse_dynamics', 'tikhonov', 'unit_var', 'TK+ID' ],
                        help = 'Additional regularization type on encoding')
    parser.add_argument('--latent_prior_weight', type=float, #default=0.05,
                        help = 'Loss weight on latent prior regularization')
    parser.add_argument('--use_psi_for_IR', action='store_true', 
                        help = 'Whether to use a separate encoding psi(obs) for IR calcs')
    parser.add_argument('--psi_enc_dim', type=int, default=64,
                        help = 'Encoder dimensionality when using a separate psi network')
    parser.add_argument('--psi_forward_loss_coef', type=float, default=1.0,
                        help = 'Loss weight on forward dynamics predictor with psi network')
    ###
    parser.add_argument('--device', type=int, help='Cuda device ID (e.g., 1)')
    args = parser.parse_args()
    if args.c_T is None:
        assert (args.c_R == 1.)
        args.c_T = args.discount
    assert (args.c_R <= 1. and args.c_T <= 1.)
    if args.domain_name == "MountainCarContinuous-v0" or args.domain_name == "Pendulum-v0":
        print('Ignoring sparsity factor for', args.domain_name)
        args.sparsity_factor = 1.0

    ### Task-specific Defaults ###
    _defaults_w = { #'Pendulum-v0' : 
                    #    { 'IRW' : 10.00, 'IRM' : 5.00, 'LPW_ID' : 0.5,  'LPW_TK' : 0.0005, 'LPW_UV' : 0.001 },
                    'SparsePendulum-v0' : 
                        { 'IRW' : 0.10, 'IRM' : 0.10, 'LPW_ID' : 0.1,  'LPW_TK' : 0.0005, 'LPW_UV' : 0.001 },
                    'ContinuousCartpole-v0' : 
                        #{ 'IRW' : 2.00, 'IRM' : 5.00, 'LPW_ID' : 1.0, 'LPW_TK' : 0.0005, 'LPW_UV' : 0.001 },
                        { 'IRW' : 2.00, 'IRM' : 0.10, 'LPW_ID' : 1.0, 'LPW_TK' : 0.0005, 'LPW_UV' : 0.001 },
                    'MountainCarContinuous-v0' :  
                        # { 'IRW' : 25.0,  'IRM' : 5.0, 'LPW_ID' : 1.00, 'LPW_TK' : 0.0005, 'LPW_UV' : 0.001 }, }
                        { 'IRW' : 20.0, 'IRM' : 0.10, 'LPW_ID' : 20.0, 'LPW_TK' : 0.005, 'LPW_UV' : 0.001 }, }
    if args.intrinsic_reward_weight is None and args.intrinsic_reward_type != 'none': 
        args.intrinsic_reward_weight = _defaults_w[args.domain_name]['IRW']
    if args.intrinsic_reward_max    is None and args.intrinsic_reward_type != 'none':
        args.intrinsic_reward_max    = _defaults_w[args.domain_name]['IRM']
    if args.use_psi_for_IR:
        print('Using separate network for IR encodings')
        assert args.intrinsic_reward_type != 'none'
        assert args.latent_prior in ['TK+ID', 'inverse_dynamics']
    if args.latent_prior_weight     is None and args.latent_prior not in ['none', None]:
        _key = { 'inverse_dynamics' : 'LPW_ID', 'tikhonov' : 'LPW_TK', 
                 'unit_var' : 'LPW_UV', }
        if args.latent_prior == 'TK+ID':
            _DW = _defaults_w[args.domain_name]
            args.latent_prior_weight = { 'LPW_ID' : _DW[ 'LPW_ID' ], 'LPW_TK' : _DW[ 'LPW_TK' ] }
        else:
            args.latent_prior_weight = _defaults_w[args.domain_name][ _key[args.latent_prior] ]
    # Whether to anneal out intrinsic rewards
    args.apply_IR_decay = False # (args.domain_name == 'MountainCarContinuous-v0')
    args.start_IR_decay = 20000
    args.end_IR_decay   = 50000
    
    #----------------------------#
    print(args)
    return args


def evaluate(env, agent, video, num_episodes, L, step, device=None, 
             embed_viz_dir=None, do_carla_metrics=None,):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0

    # embedding visualization
    obses = []
    values = []
    embeddings = []
    episode_rewards = []
    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs, torch_mode = False)

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), 
                                                   torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, info = env.step(action)

            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            video.record(env)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)

        video.save('%d.mp4' % step)
        episode_rewards.append(episode_reward)

    L.log('eval/episode_reward', np.mean(episode_rewards), step)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


def make_agent(obs_shape, action_shape, action_max, args, device):
    if args.agent == 'baseline':
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_action_max=action_max,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            encoder_max_norm=args.encoder_max_norm
        )
    elif args.agent == 'bisim':
        agent = BisimAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_action_max=action_max,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef,
            bisim_dist=args.bisim_dist,
            latprior=args.latent_prior,
            w_latprior=args.latent_prior_weight,
            c_R=args.c_R,
            c_T=args.c_T,
            build_psi=args.use_psi_for_IR,
            psi_enc_dim=args.psi_enc_dim,
            psi_forward_loss_coef=args.psi_forward_loss_coef,
            encoder_max_norm=args.encoder_max_norm
        )
    elif args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_action_max=action_max,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    if args.domain_name == 'carla' or args.domain_name in ['carla098', 'carla096']:
        print(f'Constructing Carla domain ({args.domain_name})')
        #raise NotImplementedError('Disabled carla support temporarily')
        print(f'\targs: render? {args.render}, rl_img_size? {args.image_size}')
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?
        print('\tDone')
        eval_env = env
        is_carla_task = True
    elif args.domain_name in gym.envs.registry.env_specs.keys():
        if args.sparsity_factor < 1.: # TODO a bit dangerous, since we let the sparsity_factor mean different things for different tasks
            print('Building train env with sparsity_factor {args.sparsity_factor}')
            env = gym.make(args.domain_name, sparsity_factor=args.sparsity_factor)
        else:
            print('Building train env (sparsity_factor not used)')
            env = gym.make(args.domain_name)
        print('Building eval env (sparsity_factor not used)')
        eval_env = gym.make(args.domain_name)
        if args.encoder_type == 'pixel':
            env      = wrappers.RGBArrayAsObservationWrapper(env)
            eval_env = wrappers.RGBArrayAsObservationWrapper(eval_env)
        elif args.noisy_observation:
            print('Wrapping train and eval env in noise distractor wrapper')
            env = wrappers.NoisyObservationWrapper(env, args.noisy_dims, args.noise_std)
            eval_env = wrappers.NoisyObservationWrapper(eval_env, args.noisy_dims, args.noise_std)
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=(args.encoder_type == 'pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat
        )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:%s' % args.device)

    # the dmc2gym wrapper standardizes actions
    assert abs(env.action_space.low.min()) == env.action_space.high.max()
    action_max = env.action_space.high.max()
    
    assert args.num_updates >= 1

    ###
    use_IR     = not args.intrinsic_reward_type == 'none'
    IR_type    = args.intrinsic_reward_type if use_IR else None
    w_IR       = args.intrinsic_reward_weight
    IR_max     = args.intrinsic_reward_max
    latprior   = args.latent_prior
    w_latprior = args.latent_prior_weight
    if use_IR: 
        assert w_IR > 0.0, "Intrinsic reward applied with zero weight"
        if IR_type == 'forward_dist': assert args.transition_model_type == 'probabilistic'
        print(f'Using Intrinsic Reward ({IR_type}): weight = {w_IR}')
    if not latprior == 'none': 
        if type(w_latprior) is dict:
            for k in w_latprior: assert w_latprior[k] > 0.0
        else:
            assert w_latprior > 0.0, "Latent prior loss applied with zero weight"
        print(f'Using latent prior: {latprior} with weight {w_latprior}')
    ###
    
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device
    )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_max=action_max,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    if args.num_updates != 1:
        print(f'Warning! Using num_updates = {args.num_updates} > 1 with a single env!')

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0:
                print('Running eval: episode', episode) 
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step, 
                         do_carla_metrics = False,) # is_isaac_task = False) #is_carla_task)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, torch_mode = False)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else args.num_updates
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        curr_reward = reward # curr_reward = reward from prev iter
        next_obs, reward, done, _ = env.step(action)

        if use_IR: 
            scaler = 1.0
            if step >= args.end_IR_decay and args.apply_IR_decay:
                intrinsic_reward = 0.0
                scaler = 0.0
            else:
                # Whether to normalize IR with mu_bd
                NORM_IR_BY_MU_BD = args.norm_ir_by_mu_bd
                # TODO ensure works with isaac
                intrinsic_reward = agent.compute_curiosity_reward(obs, next_obs, action, 
                                        IR_type, IR_max, normalize_by_mu_bd = NORM_IR_BY_MU_BD)
                if args.apply_IR_decay and step >= args.start_IR_decay:
                    scaler = 1.0 - ( (step - args.start_IR_decay) / (args.end_IR_decay - args.start_IR_decay) )
            if step % 1000 == 0:
                print(step, f'IR: {intrinsic_reward}, wsIR: {w_IR * scaler * intrinsic_reward}, ER: {reward}, (scaler: {scaler})')
            intrinsic_reward = w_IR * scaler * intrinsic_reward
            L.log('train/ir', intrinsic_reward, step)
            reward += intrinsic_reward
        # </IR/> #

        # allow infinite bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        # Curr_reward = previous iteration reward
        # Reward = reward after taking the current action [r(s_t, a_t)]
        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)
        episode_step += 1
        
        obs = next_obs


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    main()
