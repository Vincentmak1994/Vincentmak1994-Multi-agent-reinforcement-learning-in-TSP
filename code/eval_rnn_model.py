import torch 
from dqn_network import DQN, DQN_RNN
from sensor_network import Network
from itertools import count
import torch.nn.functional as F
from agent import Agent
from tqdm import tqdm
import os
import csv

def load_model(path, hidden_1, hidden_2, name=None):
    checkpoint = torch.load(path)
    policy_net = DQN_RNN(num_cities, hidden_1, hidden_2)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    return policy_net

budgets = [5000, 6000, 7000]
episodes= [2000, 3000, 5000, 10000]
test_number=4

for episode in episodes:
    for budget in budgets:
        for city in range(10):
            file_path = f'project/code/model/rnn/city_10/test_4/episode_{episode}/model_size_64_128/batch_32_lr_0.001_memoryBuffer_200000/starting_city_{city}/budget_{budget}/best_model.tar'
            hidden_size_1 = 64
            hidden_size_2 = 128
            pred = {}

            network = Network(can_revisit=False).build_city_sample(start_city=city, unit='mile')
            num_cities = network.num_node
            model = load_model(file_path, hidden_size_1, hidden_size_2)
            agent = Agent(None, num_cities, model, model, None, lr=0.0001, gamma=None, agent_name=None, budget=6000, is_training=False)

            path = [city]

            for timestep in count():
                state = network.current_nodes(one_hot_encoding=False)
                state = F.one_hot(torch.tensor([state]), num_classes=num_cities).to(torch.float32)

                state_location = network.state_location()
                state_location = torch.tensor(state_location, dtype=torch.float32).unsqueeze(0)

                feasible_mask = network.get_feasible_mask(agent.current_budget())
                feasible_mask = torch.tensor(feasible_mask, dtype=torch.float32).view(1, 1, -1)
                
                action = agent.select_action(state, state_location, feasible_mask)
                path.append(action)

                next_state_representation, reward, is_done, next_state_location, cost = network.visit(action)
                agent.collect_prize_n_adjust_budget(reward, cost)

                if is_done:
                    if action != city:
                        cost_to_starting = network.min_cost_graph[network.current_nodes()]['min_cost']
                        agent.collect_prize_n_adjust_budget(0, cost_to_starting)
                        path.append(city) #travel back to starting city
                    break 
            pred['starting_city'] = city
            pred['total_episode'] = episode
            pred['total_budget'] = budget
            pred['reward'] = agent.collected_prizes()
            pred['path'] = path
            
            pred_detail = f'project/code/model/rnn/city_{num_cities}/test_{test_number}'
            os.makedirs(pred_detail, exist_ok=True)
            pred_detail_file = f'{pred_detail}/prediction_data.csv'
            training_file_exists = os.path.isfile(pred_detail_file)
            with open(pred_detail_file, 'a', newline='') as csvfile:
                fieldnames = pred.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not training_file_exists:
                    writer.writeheader()
                writer.writerow(pred)







'''

# sample_network = Network(can_revisit=False).build_city_sample(start_city=0, unit='mile')
start_city=0

print("====================================================")
print("====================================================")
print("====================================================")

network = Network(can_revisit=False).build_city_sample(start_city=start_city, unit='mile')
num_cities = network.num_node
model = load_model(hidden_size_1, hidden_size_2)
agent = Agent(None, num_cities, model, model, None, lr=0.0001, gamma=None, agent_name=None, budget=6000, is_training=False)
for test in tqdm(range(10)):
    path = [start_city]
    agent.reset_budget_n_prize()
    
    network.reset()
    for timestep in count():
        state = network.current_nodes(one_hot_encoding=False)
        state = F.one_hot(torch.tensor([state]), num_classes=num_cities).to(torch.float32)

        state_location = network.state_location()
        state_location = torch.tensor(state_location, dtype=torch.float32).unsqueeze(0)

        feasible_mask = network.get_feasible_mask(agent.current_budget())
        feasible_mask = torch.tensor(feasible_mask, dtype=torch.float32).view(1, 1, -1)
        
        action = agent.select_action(state, state_location, feasible_mask)
        path.append(action)

        next_state_representation, reward, is_done, next_state_location, cost = network.visit(action)
        agent.collect_prize_n_adjust_budget(reward, cost)

        if is_done:
            if action != start_city:
                cost_to_starting = network.min_cost_graph[network.current_nodes()]['min_cost']
                agent.collect_prize_n_adjust_budget(0, cost_to_starting)
                path.append(start_city) #travel back to starting city
            break 
    print(f"pred_{test+1} at city {start_city}: {agent.collected_prizes()} Pred path: {path}\n")



hidden_size_1 = 64
# [64, 128]
hidden_size_2 = 128
path  = 'project/code/model/city_10/test_1/model_size_64_128/batch_64_lr_0.001_memoryBuffer_200000/starting_city_0/simple_dqn.tar'
# 'project/code/model/simple_dqn.tar'

checkpoint = torch.load(path)

policy_net = DQN(10, 10, hidden_size_1, hidden_size_2)
print(f"before: {policy_net.state_dict()}")

policy_net.load_state_dict(checkpoint['model_state_dict'])
print(f"after: {policy_net.state_dict()}")

# print(policy_net.eval())
'''