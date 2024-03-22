import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_learning_curve, get_learning_curve_data

for city in range(2):
    rnn_df = pd.read_csv(f'project/code/model/rnn/city_48/test_10/budget_10000/starting_city_{city}/total_ep_5010/num_test_0/training_data_each_episode.csv')

    'project/code/model/rnn/city_10/test_7/budget_6000/starting_city_0/total_ep_5000/num_test_9/training_data_each_episode.csv'
    'project/code/model/rnn/city_10/test_5/budget_6000/starting_city_0/training_data_each_episode.csv'
    episode = rnn_df['episode']
    rnn_training_reward = rnn_df['reward']
    rnn_running_avg_data = get_learning_curve_data(episode, rnn_training_reward)

    marl_v2_df = pd.read_csv('project/code/model/marl_v2/test_5/ep_5000_5000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv')
    marl_v2_df = marl_v2_df[marl_v2_df['total_budget'] == 6000]
    episode = marl_v2_df['episode']
    marl_v2_training_reward = marl_v2_df['max_reward']
    marl_v2_running_avg_data = get_learning_curve_data(episode, marl_v2_training_reward)

    marl_v1_df = pd.read_csv('project/code/model/marl_v1/test_5/ep_5000_5000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv')
    marl_v1_df = marl_v1_df[marl_v1_df['total_budget'] == 6000]
    episode = marl_v1_df['episode']
    marl_v1_training_reward = marl_v1_df['max_reward']
    marl_v1_running_avg_data = get_learning_curve_data(episode, marl_v1_training_reward)


    plt.plot(rnn_running_avg_data[0], rnn_running_avg_data[1], label='rnn', color='orange')
    plt.axhline(y=479, label='optimal', color='g', linestyle='-')
    plt.plot(marl_v1_running_avg_data[0], marl_v1_running_avg_data[1], label='marl_v1', color='r')
    plt.plot(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1], label='marl_v2', color='b')
    plt.legend()
    plt.title(f'Training Reward')
    plt.xlabel('episode')
    plt.ylabel('reward')
    # plt.savefig('training_reward_comparison.png')
    plt.show()




# plot_learning_curve(episode, rnn_training_reward, 'training_plot.png', 'marl v1 training reward', 'running average reward')
# print(len(marl_v2_training_reward))


# cols = 2
# rows = 4 // cols 
# fig, ax = plt.subplots(rows, cols, figsize=(10,10))

