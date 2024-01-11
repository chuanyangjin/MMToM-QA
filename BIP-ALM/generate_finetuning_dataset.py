import os
import json
import sys
import random
import pickle
import argparse
random.seed(42)
sys.path.append('..')
from graph_utils import TEST_INDICES, filter_graph, get_id2node, transform_graph_training

def load_pickles(path, folder=True):
    if folder:
        pickle_data = {}
        for file in os.listdir(path):
            if file.endswith(".pik"):
                with open(os.path.join(path, file), 'rb') as f:
                    data = pickle.load(f)
                    pickle_data[file] = data
    else:
        if path.endswith(".pik"):
            with open(path, 'rb') as f:
                pickle_data = pickle.load(f)
    return pickle_data

def split_actions(actions):
    # Extract verbs and nouns from actions
    verbs = []
    nouns = []
    noun_ids = []
    for t in range(len(actions)):
        action = actions[t]
        verb, noun, noun_id = action.split(" ")[0][1:-1], action.split(" ")[1][1:-1], action.split(" ")[2][1:-1]
        verbs.append(verb)
        nouns.append(noun)
        noun_ids.append(noun_id)
    return verbs, nouns, noun_ids

def transform_actions(actions):
    transformed_actions = []
    for action in actions:
        transformed_actions.append(action.split(" ")[0][1:-1] + ' ' +  action.split(" ")[1][1:-1])
    return transformed_actions

def transform_action(action, new_names = {}, symbolic = False):
    verb = action.split(" ")[0][1:-1]
    if int(action.split(" ")[2][1:-1]) in new_names:
        noun = new_names[int(action.split(" ")[2][1:-1])]
    else:
        noun = action.split(" ")[1][1:-1]

    if not symbolic:
        return verb + ' ' +  noun
    else:
        return verb + ' (' +  noun + ')'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='finetuning_data')
    parser.add_argument("--output_file", type=str, default='formatted_finetuning_data.json')
    args = parser.parse_args()

    vqa_dataset = []
    data_path = args.input_data
    episodes = list(set(range(1200)) - set(TEST_INDICES))

    for episode in episodes:
        # Load task data
        task_path = f'{data_path}/task_{episode}/'        
        try:
            task = load_pickles(task_path, folder=True)
        except Exception as e:
            print(f'Error loading task from {task_path}: {e}')
            continue
        num_time = sum(1 for file_name in os.listdir(f'{data_path}/task_{episode}') if file_name.startswith('graph_') and file_name.endswith('.pik'))
        
        # Obtain the initial state
        init_graph = task['init_graph.pik']
        init_graph = filter_graph(init_graph)
        id2node = get_id2node(init_graph)
    
        # Obtain the goal
        try:
            last_action = task['actions.pik'][-1]
        except Exception as e:
            print(f"Missing action {episode}.pik")
            continue

        goal = last_action.split(" ")[1][1:-1]
        goal_id = last_action.split(" ")[2][1:-1]

        pickle_data = task['env_info.pik']
        env_id = pickle_data['env_id']

        # Differentiate between multiple kitchencabinets
        kitchencabinets = {}
        for id, node in id2node.items():
            if node['class_name'] == 'kitchencabinet':
                kitchencabinets[id] = node['obj_transform']['position']
        
        if env_id == 0:
            kitchencabinets = dict(sorted(kitchencabinets.items(), key=lambda x: x[1][0])) # from down to up
        elif env_id in [1, 4]:
            kitchencabinets = dict(reversed(sorted(kitchencabinets.items(), key=lambda x: x[1][0]))) # from up to down
        elif env_id in [2, 3]:
            kitchencabinets = dict(sorted(kitchencabinets.items(), key=lambda x: x[1][2])) # from left to right

        new_values = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th']
        kitchencabinets = {k: f'{new_values[i]} kitchencabinet' for i, (k, v) in enumerate(kitchencabinets.items())}
        id2name = {id: kitchencabinets[id] if id in kitchencabinets else node["class_name"] for id, node in id2node.items()}
        
        # record belief
        beliefs = {}
        for time in range(num_time):
            belief_data = task[f'belief_{time}.pik']['belief'][int(goal_id)]
            belief_distribution = {}
            for relation in ['INSIDE', 'ON']:
                for i in range(1, len(belief_data[relation][0])):
                    id = belief_data[relation][0][i]
                    if id in id2name:
                        location = id2name[belief_data[relation][0][i]]
                        if belief_data[relation][1][i] > -10000:
                            belief_distribution[location] = belief_data[relation][1][i]

            # beliefs[time] = result_string = ', '.join(sorted(belief_distribution, key=lambda x: belief_distribution[x], reverse=True))
            keys = list(belief_distribution.keys())
            random.shuffle(keys)
            beliefs[time] = result_string = ', '.join(keys)


        # Collect training data
        actions = task['actions.pik']
        for question_time in range(num_time):
            graph = task[f'graph_{question_time}.pik']
            graph = filter_graph(graph)
            graph = transform_graph_training(graph, kitchencabinets)

            # Predict action: G, S_t, B_t → A_t
            vqa = {}
            current_action = transform_action(actions[question_time], kitchencabinets)
            vqa['input'] = f"goal: {goal} \nWhat's inside the apartment: {graph} \nbelief (possible locations the person suspects the {goal} could be): {beliefs[question_time]} \naction: "
            if len(vqa_dataset) != 0 and vqa['input'] == vqa_dataset[-1]['input']: continue
            vqa['ref'] = f"{current_action}"
            vqa_dataset.append(vqa)

            # # Predict belief: G, S_t, B_t, A_t → B_{t+1}
            # if question_time != num_time - 1:
            #     vqa = {}
            #     vqa['input'] = f"goal: {goal} \nstate: {graph} \nbelief (a list of potential locations of {goal}): {beliefs[question_time]} \naction: {current_action} \nnext belief (a list of potential locations of {goal}): "
            #     vqa['ref'] = f"{beliefs[question_time + 1]}"
            #     vqa_dataset.append(vqa)
        

    print(len(vqa_dataset))
    sampled_vqa_dataset = random.sample(vqa_dataset, 20000)

    # Store json file    
    file_path = args.output_file
    with open(file_path, "w") as file:
        pass
    for vqa in sampled_vqa_dataset:
        with open(file_path, 'a') as file:
            json.dump(vqa, file)
            file.write('\n')