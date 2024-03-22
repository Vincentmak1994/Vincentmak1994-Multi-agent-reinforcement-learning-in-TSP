import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

opt_solution = [479,481,495,505,514,529,536,542,551,551]
version = 'v2'
path = f"project/code/model/marl_{version}/test_2/plots"
os.makedirs(path, exist_ok=True)

training_df = pd.read_csv(f'project/code/model/marl_{version}/test_2/ep_2000_10000/budget_6000_6000/num_agent_3_3/training/marl_10_city.csv')
training_df_details = training_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['max_reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])
training_df_details = training_df_details.drop(columns=['num_agent'])
training_df_details = training_df_details.rename(columns={'max_reward':'training_max_reward'})
training_df_details = training_df_details.set_index(['starting_city',  'total_episode', 'total_budget'])

prediction_df = pd.read_csv(f'project/code/model/marl_{version}/test_2/ep_2000_10000/budget_6000_6000/num_agent_3_3/execution/marl_10_city.csv')
prediction_df_details = prediction_df.groupby(['starting_city', 'num_agent', 'total_episode', 'total_budget'])['reward'].max().reset_index().sort_values(by=['starting_city', 'num_agent', 'total_episode', 'total_budget'])
prediction_df_details = prediction_df_details.drop(columns=['num_agent'])
prediction_df_details = prediction_df_details.rename(columns={'reward':'prediction_reward'})
prediction_df_details = prediction_df_details.set_index(['starting_city',  'total_episode', 'total_budget'])

marl_df = pd.concat([training_df_details, prediction_df_details], axis=1, join='inner').reset_index()


cols = 2
rows = 10 // cols 
fig, ax = plt.subplots(rows, cols, figsize=(10,10))

for city in range(10):
    fig.suptitle(f'MARL {version}')
    axs = ax[city// cols, city%cols]
    episodes = marl_df[marl_df['starting_city'] == city]['total_episode']
    barWidth = 0.20
    br1 = np.arange(len(episodes)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    # # br4 = [x + barWidth for x in br3] 
    # # br5 = [x + barWidth for x in br4] 
    # # br6 = [x + barWidth for x in br5] 
    axs.bar(br1, opt_solution[city], color='g', width=barWidth, edgecolor ='grey', label ='optimal_reward')
    # plt.bar(br2, rnn_training, color='b', width=barWidth, edgecolor ='grey', label ='training_max_reward')
    # plt.bar(br3, rnn_pred, color='r', width=barWidth, edgecolor ='grey', label ='prediction_reward')
    axs.bar(br2, marl_df[marl_df['starting_city'] == city]['training_max_reward'], color='b', width=barWidth, edgecolor ='grey', label =f'training')
    axs.bar(br3, marl_df[marl_df['starting_city'] == city]['prediction_reward'], color='r', width=barWidth, edgecolor ='grey', label ='prediction')

    axs.set_xlabel('episode')
    axs.set_ylabel('reward')
    # plt.title('Starting City: 0 - RNN performance comparison')
    axs.set_title(f'Starting City: {city}')
    axs.set_xticks([r + barWidth for r in range(len(episodes))], 
            episodes)
    if city == 0:
        axs.legend(bbox_to_anchor=(2.5, 2.3))
    plt.subplots_adjust(wspace=0.4,hspace=1)


fig.savefig(f"{path}/marl_{version}_training_vs_prediction.png") 
plt.show()