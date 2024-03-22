import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

optimal = [210, 479, 498, 498]
marl_v1 = [200, 442, 498, 498]
# [210, 399,498, 498]
marl_v2 = [210, 399, 498, 498]
# [210, 145, 455, 498]
rnn = [210, 442, 498, 498]
budgets = [2000, 6000, 10000, 15000]

barWidth = 0.20
br1 = np.arange(len(budgets)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
# # br5 = [x + barWidth for x in br4] 
# # br6 = [x + barWidth for x in br5] 

plt.bar(br1, optimal, color='g', width=barWidth, edgecolor ='grey', label ='optimal')
# plt.bar(br2, rnn_training, color='b', width=barWidth, edgecolor ='grey', label ='training_max_reward')
# plt.bar(br3, rnn_pred, color='r', width=barWidth, edgecolor ='grey', label ='prediction_reward')
plt.bar(br2, marl_v1, color='b', width=barWidth, edgecolor ='grey', label ='marl_v1')
plt.bar(br3, marl_v2, color='r', width=barWidth, edgecolor ='grey', label ='marl_v2')
plt.bar(br4, rnn, color='c', width=barWidth, edgecolor ='grey', label ='rnn')

plt.xlabel('budgets')
plt.ylabel('rewards')
# plt.title('Starting City: 0 - RNN performance comparison')
plt.title('Starting City: 0 - Reward Collected (execution)')
plt.xticks([r + barWidth for r in range(len(budgets))], 
        budgets)
plt.legend()
plt.savefig("marl_vs_rnn_execution_with_c-table.png") 
plt.show()

