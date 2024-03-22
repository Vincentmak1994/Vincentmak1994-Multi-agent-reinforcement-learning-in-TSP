import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

budgets = [10000, 20000, 30000, 40000]
rnn_reward = [998, 1796, 2002, 2452]
marl_reward = [1797, 2452, 2452, 2452]
rnn_remaining = [6.021855, 62.375591, 61.525174, 1423.994076]
marl_remaining = [22.801456, 5691.950926, 15691.950926, 25691.950926]

rnn_time = [143.18264913558957, 254.8848416805267, 301.24610447883606, 468.2416787147522]
marl_time = [9.12596154212949, 11.524316787719703, 13.16286849975584, 13.967457056045516 ]

barWidth = 0.20
br1 = np.arange(len(budgets)) 
br2 = [x + barWidth for x in br1] 
# br3 = [x + barWidth for x in br2] 
# br4 = [x + barWidth for x in br3] 
# # br5 = [x + barWidth for x in br4] 
# # br6 = [x + barWidth for x in br5] 



'''
plt.bar(br1, rnn_reward, color='b', width=barWidth, edgecolor ='grey', label ='RNN')
plt.bar(br2, marl_reward , color='r', width=barWidth, edgecolor ='grey', label ='P-MARL')
plt.xlabel('Budgets', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.xticks([r + barWidth for r in range(len(budgets))], 
        budgets, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize="15")
plt.tight_layout()
plt.savefig("marl_vs_rnn_all_cities.png") 
plt.show()


plt.bar(br1, [a-b for a,b in zip(budgets,rnn_remaining)], color='b', width=barWidth, edgecolor ='grey', label ='RNN')
plt.bar(br2, [a-b for a,b in zip(budgets,marl_remaining)] , color='r', width=barWidth, edgecolor ='grey', label ='P-MARL')
plt.xlabel('Budgets', fontsize=15)
plt.ylabel('Distance', fontsize=15)
plt.xticks([r + barWidth for r in range(len(budgets))], 
        budgets, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize="15")
plt.tight_layout()
plt.savefig("marl_vs_rnn_all_cities_distance.png") 
plt.show()



'''
plt.bar(br1, rnn_time, color='b', width=barWidth, edgecolor ='grey', label ='RNN')
plt.bar(br2, marl_time , color='r', width=barWidth, edgecolor ='grey', label ='P-MARL')
plt.xlabel('Budgets', fontsize=15)
plt.ylabel('Time in second', fontsize=15)
plt.xticks([r + barWidth for r in range(len(budgets))], 
        budgets,fontsize=15)
plt.yticks(fontsize=15)        
plt.legend(fontsize="15")
plt.tight_layout()
plt.savefig("marl_vs_rnn_all_cities_time.png") 
plt.show()