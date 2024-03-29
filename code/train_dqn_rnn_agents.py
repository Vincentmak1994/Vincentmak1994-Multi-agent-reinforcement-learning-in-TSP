from utils import EpsilonGreedyStrategy
import torch 
import os
from sensor_network import Network
from dqn_network import DQN, DQN_RNN
from agent import Agent 
from memory_buffer import ReplayMemory
from training_loop import training_loop

num_episodes = 5000
test_number = 3
batch_size = 32
gamma = 0.999   #discount rate
eps_start = 1   #Starting epsilon (for epsilon greedy strategy)
eps_end = 0.05
eps_decay = 2500
target_update = 10 #how frequent to update weights of the target network 
memory_size = 200000 #replay memory capacity 
lr = 0.001       #learning rate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
hidden_size_1 = [64, 128]
hidden_size_2 = [128, 256]
model_info = {}


for hidden_1 in hidden_size_1:
    for hidden_2 in hidden_size_2:
        model_info[f"model_size_{hidden_1}_{hidden_2}"] = {} 
        city_test_info = {}

        for i in range(1):
            city_test_info[f"starting_city_{i}"] = {}
            episode_info = {}
            city_network = Network(can_revisit=False).build_city_sample(start_city=i, unit='mile')
            num_cities = city_network.num_node
        #     print("# of cities: {}\nTotal prize:{}\nCan agent revisit?:{}\nStarting city:{}\nEnding city:{}\n".format(num_cities, sum(city_network.get_prizes_array()), city_network.can_revisit, city_network.current_nodes(), city_network.end))
            print(f"************** model_size_{hidden_1}_{hidden_2} **************")
            print("=== Test {} : starting city {} ===".format(i+1, i+1))

            policy_net = DQN(n_observation=num_cities,
                             n_actions=num_cities,
                             hidden_size_1=hidden_1, 
                             hidden_size_2=hidden_2).to(device)

            target_net = DQN(n_observation=num_cities,
                             n_actions=num_cities,
                             hidden_size_1=hidden_1, 
                             hidden_size_2=hidden_2).to(device)

            rnn_policy_net = DQN_RNN(num_cities=num_cities,
                                     hidden_size_1=hidden_1, 
                                     hidden_size_2=hidden_2).to(device)
            rnn_target_net = DQN_RNN(num_cities=num_cities,
                                     hidden_size_1=hidden_1, 
                                     hidden_size_2=hidden_2).to(device)

            budget = 6000
            agent = Agent(strategy, num_cities, policy_net, target_net, ReplayMemory(memory_size), lr=lr, gamma=gamma, agent_name='simple_dqn', budget=budget)
            rnn_agent = Agent(strategy, num_cities, rnn_policy_net, rnn_target_net, ReplayMemory(memory_size), lr=lr, gamma=gamma, agent_name='rnn_dqn', budget=budget)

            # memory = ReplayMemory(memory_size)

            agents = [agent, rnn_agent]

            game_info = training_loop(city_network, num_episodes, agents, batch_size, target_update, save_model=True)
            for key in game_info.keys():
                print(f"agent:{key}, max_reward:{game_info[key]['max_reward']}, max_reward_ep:{game_info[key]['max_reward_episode']}")
            print("\n")

            city_test_info[f"starting_city_{i}"] = game_info

        model_info[f"model_size_{hidden_1}_{hidden_2}"] = city_test_info

opt_solution = [479,481,495,505,514,529,536,542,551,551]
test_file = f'project/code/model/city_{num_cities}/test_{test_number}'
os.makedirs(test_file, exist_ok=True)
with open(f'{test_file}/summary.txt', 'w') as file:
    for model_size in model_info.keys():
        file.write("===========================================================================\n")
        file.write(f"====================== model_size: {model_size} ======================\n")
        file.write("===========================================================================\n")
        # print(f"******** model_size: {model_size} ********\n")
        city = 0 
        for starting_city in model_info[model_size].keys():
            file.write(f"Starting City: {starting_city}\n")
            # print(f"Starting City: {starting_city}\n")
            for model, data in model_info[model_size][starting_city].items():
                file.write(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}, is_optimal: {opt_solution[city] == data['max_reward']}\n")
                # print(f"Model_name: {model}, max_reward: {data['max_reward']}, best_path: {data['best_path']}, is_optimal: {opt_solution[city] == data['max_reward']}")
            city += 1
            file.write("\n")


for model_size in model_info.keys():
    print(f"******** saving model_size: {model_size} ********\n")
    city = 0 
    for starting_city in model_info[model_size].keys():
        for model, data in model_info[model_size][starting_city].items():
            path = f'{test_file}/{model_size}/batch_{batch_size}_lr_{lr}_memoryBuffer_{memory_size}/starting_city_{city}/'
            os.makedirs(path, exist_ok=True)
            filename = f"{model}.tar"
            torch.save({
                'epoch': data['best_model_weights']['epoch'],
                'model_state_dict': data['best_model_weights']['model_state_dict'],
                'optimizer_state_dict': data['best_model_weights']['optimizer_state_dict'],
                # Add any other relevant information
            }, path+'/'+filename)
        city += 1
