# MIT License

# Copyright (c) 2024 Feiyu Yang 杨飞宇

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
from chenglongchenMAA2C import MAA2C
from common.utils import agg_double_list,addScalarFcn,globalEpisode
import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt

from RayMultiAgentEnv import MyEnvironmentRay

MAX_EPISODES = 5000

EVAL_EPISODES = 5
EVAL_INTERVAL = 25

RANDOM_SEED = 2024

def run(env_id="SingleTrackRailNetTrainScheduling-v0"):
    env = MyEnvironmentRay(device='cuda')
    env_eval = MyEnvironmentRay(device='cuda')
    state_dims = [58,58,58,82,82,82,82,58,58,58,74,58,58,58,58,58,58,74]
    action_dims = [7, 7, 7, 10, 10, 10, 10, 7, 7, 7, 9, 7, 7, 7, 7, 7, 7, 9]
    n_agents=18
    maa2c = MAA2C(env, n_agents,
                 memory_capacity=1000, max_steps=None,
                 roll_out_n_steps=20,
                 reward_gamma=0.96, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=256,
                 actor_output_act=nn.functional.log_softmax, critic_loss="huber",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.1,
                 max_grad_norm=10, batch_size=512, episodes_before_train=5,
                 epsilon_start=0.05, epsilon_end=0.05, epsilon_decay=2000,
                 use_cuda=True, training_strategy="centralized",
                 actor_parameter_sharing=False, critic_parameter_sharing=True)
    episodes =[]
    eval_rewards =[]

    while maa2c.n_episodes < MAX_EPISODES:
        globalEpisode[0]+=1
        with torch.no_grad():
            maa2c.interact()
        maa2c.train()
        if maa2c.episode_done:
            print(f'n_episodes:{maa2c.n_episodes}')
        if maa2c.episode_done and ((maa2c.n_episodes-1)%EVAL_INTERVAL == 0):
            with torch.no_grad():
                rewards, _ = maa2c.evaluation(env_eval, EVAL_EPISODES)
                rewards_mu, rewards_std = agg_double_list(rewards)
                for idx in range(maa2c.n_agents):
                    addScalarFcn(f'Evaluation_Reward_mu/Agent{idx}',rewards_mu[idx],maa2c.n_episodes-1)
                    addScalarFcn(f'Evaluation_Reward_std/Agent{idx}',rewards_std[idx],maa2c.n_episodes-1)
                print("Episode %d, Average Ep Reward of All Agents%.2f" % (maa2c.n_episodes-1, sum(rewards_mu)))
                episodes.append(maa2c.n_episodes-1)
                eval_rewards.append(sum(rewards_mu)/len(rewards_mu))
                addScalarFcn('Average Evaluation Reward of All Agents',sum(rewards_mu),maa2c.n_episodes-1)
        if maa2c.episode_done and ((maa2c.n_episodes-1)%500 == 0):
            torch.save(maa2c,f'maa2c_{maa2c.n_episodes}.pth')
    


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run(sys.argv[1])
    else:
        run()