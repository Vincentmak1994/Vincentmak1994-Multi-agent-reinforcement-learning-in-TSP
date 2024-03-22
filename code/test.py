'''
import re
f = 'project/code/model/rnn/city_10/test_2/episode_5000/summary.txt'

d = {}
start = 0 
with open(f) as file:
    lines = [line.rstrip() for line in file]

# print(lines.split(','))
for line in lines:
    temp = {}
    item = line.split(',')
    for element in item:
        if re.match(r"(^[ a-zA-z_](.)*): (.)+", element):
            # print(element)
            key, val = element.split(':')[0], element.split(':')[1]
            temp[key] = val
    if len(temp) != 0:
        d[start] = temp
        start += 1
print(d)
#         print(item.split(':')) 
'''
# import pandas as pd 
# marl_v2_training_df = pd.read_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training/marl_10_city.csv')
# marl_v2_training_df['starting_city'] = pd.Series()
# for i in range(len(marl_v2_training_df)):
#     marl_v2_training_df['starting_city'][i] = marl_v2_training_df['max_reward_path'][i].replace('[','').replace(']','')[0]

# marl_v2_training_df.to_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/training/marl_10_city_2.csv', index=False)

'''
marl_v2_pred = pd.read_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/execution/marl_10_city.csv')
marl_v2_pred['total_episode'] = pd.Series()
ep = [2000, 3000, 5000, 10000]
start = -1
for i in range(len(marl_v2_pred)):
    if i % 120 == 0:
        start = -1

    if i % 30 == 0:
        start += 1 
    marl_v2_pred['total_episode'][i] =  int(ep[start])

# print(marl_v2_pred[marl_v2_pred['total_episode'] == 2000])
marl_v2_pred.to_csv('project/code/model/marl_v1/test_1/ep_2000_10000/budget_5000_7000/num_agent_3_10/execution/marl_10_city_2.csv', index=False)

# print(len(marl_v2_pred))

'''
# [2000, 3000, 5000, 10000]

# d = {0: {'a':1, 'b':2}}

# with open('project/code/Capital_Cities.txt', 'r') as f:
#         if len(line.strip()) != 0:
#            item =line.split()
#            city, x, y , prize = item[0], float(item[1]), float(item[2]), int(item[3])
#            print((city, x, y, prize))


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import plot_learning_curve, get_learning_curve_data

rnn_df = pd.read_csv('project/code/model/rnn/city_48/test_9.1/budget_40000/starting_city_0/total_ep_5010/num_test_0/training_data_each_episode.csv')

print(sum(rnn_df['execution_time']))

# marl = pd.read_csv('project/code/model/marl_v2/test_9/ep_5000_5000/budget_10000_40000/num_agent_3_3/training/marl_10_city.csv')
# print(sum(marl[(marl['total_budget'] == 40000)]['execution_time']))

'''
episode = rnn_df['episode']
rnn_training_reward = rnn_df['reward']
rnn_training_time = rnn_df['execution_time']
rnn_running_avg_data = get_learning_curve_data(episode, rnn_training_reward)
# rnn_training_time_avg = get_learning_curve_data(episode, rnn_training_time)
# plt.plot(rnn_training_time_avg[0], rnn_training_time_avg[1])
# plt.show()

marl_v2_df = pd.read_csv('project/code/model/marl_v2/test_8.0/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv')
'project/code/model/marl_v2/test_7.4/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v2/test_7.2/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v2/test_7.1/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v2/test_3/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v2/test_7/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'
print(f'max_reward: {marl_v2_df["max_reward"].max()}')

marl_v2_df = marl_v2_df[marl_v2_df['total_budget'] == 6000]
marl_v2_df = marl_v2_df[:5000]
episode = marl_v2_df['episode']
marl_v2_training_reward = marl_v2_df['max_reward']
# marl_v2_training_time = marl_v2_df['execution_time']
marl_v2_running_avg_data = get_learning_curve_data(episode, marl_v2_training_reward)
# marl_v2__time_avg = get_learning_curve_data(episode, marl_v2_training_time)

marl_v1_df = pd.read_csv('project/code/model/marl_v1/test_7.4/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv')

'project/code/model/marl_v1/test_7.2/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v1/test_3/ep_20000_20000/budget_2000_150000/num_agent_3_3/training/marl_10_city.csv'
'project/code/model/marl_v2/test_6/ep_20000_20000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv'
marl_v1_df = marl_v1_df[marl_v1_df['total_budget'] == 6000]
marl_v1_df = marl_v1_df[:5000]
episode = marl_v1_df['episode']
marl_v1_training_reward = marl_v1_df['max_reward']
# marl_v1_training_time = marl_v1_df['execution_time']
marl_v1_running_avg_data = get_learning_curve_data(episode, marl_v1_training_reward)
# marl_v1__time_avg = get_learning_curve_data(episode, marl_v1_training_time)


plt.axhline(y=479, label='optimal', color='g', linestyle='-')
# plt.plot(rnn_training_time_avg[0], rnn_training_time_avg[1], label='rnn', color='orange')
# plt.plot(marl_v1__time_avg[0], marl_v1__time_avg[1], label='marl_v1', color='r')
# plt.plot(marl_v2__time_avg[0], marl_v2__time_avg[1], label='marl_v2', color='b')
# plt.legend()
# plt.title('Training Time Comparison - RNN vs MARL')
# plt.ylabel('time (second)')
# plt.xlabel('episode')
# plt.savefig('training_time_comparison.png')
# plt.show()


# plt.axhline(y=210, label='optimal', color='g', linestyle='-')
# plt.plot(rnn_running_avg_data[0], rnn_running_avg_data[1], label='rnn', color='orange')
plt.plot(episode, rnn_training_reward, label='rnn', color='orange')
# plt.plot(marl_v1_running_avg_data[0], marl_v1_running_avg_data[1], label='marl_v1', color='r')
plt.plot(episode, marl_v1_training_reward, label='marl_v1', color='r')
# plt.plot(marl_v2_running_avg_data[0], marl_v2_running_avg_data[1], label='marl_v2', color='b')
plt.plot(episode, marl_v2_training_reward, label='marl_v2', color='b')

plt.legend()
plt.title('Training Reward Comparison - RNN vs MARL')
plt.ylabel('reward')
plt.xlabel('episode')
plt.savefig('training_reward_comparison_8.0_first_5000ep_row.png')
plt.show()




# # plot_learning_curve(episode, rnn_training_reward, 'training_plot.png', 'marl v1 training reward', 'running average reward')
# # print(len(marl_v2_training_reward))


# # cols = 2
# # rows = 4 // cols 
# # fig, ax = plt.subplots(rows, cols, figsize=(10,10))

'''