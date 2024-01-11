import sys
import json
import random
import openai
import argparse
from tqdm import tqdm
from graph_utils import POSSIBLE_BELIEF, POSSIBLE_CONTAINER, ROOM_LIST, SURFACE_LIST
random.seed(42)
sys.path.append('..')
# openai.api_key = YOUR-OPENAI-KEY


def generate_response(prompt, max_tokens = 100, temperature = 0):
    response = openai.ChatCompletion.create(
        model = "gpt-4",    # choose in "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response["choices"][0]["message"]["content"]


def replace_words(s):
    replacements = [
            ('refrigerator', 'fridge'),
            ("first kitchen cabinet", "1st kitchencabinet"),
            ("second kitchen cabinet", "2nd kitchencabinet"),
            ("third kitchen cabinet", "3rd kitchencabinet"),
            ("fourth kitchen cabinet", "4th kitchencabinet"),
            ("fifth kitchen cabinet", "5th kitchencabinet"),
            ("sixth kitchen cabinet", "6th kitchencabinet"),
            ("seventh kitchen cabinet", "7th kitchencabinet"),
            ("eighth kitchen cabinet", "8th kitchencabinet"),
            ("first cabinet", "1st kitchencabinet"),
            ("second cabinet", "2nd kitchencabinet"),
            ("third cabinet", "3rd kitchencabinet"),
            ("fourth cabinet", "4th kitchencabinet"),
            ("fifth cabinet", "5th kitchencabinet"),
            ("sixth cabinet", "6th kitchencabinet"),
            ("seventh cabinet", "7th kitchencabinet"),
            ("eighth cabinet", "8th kitchencabinet"),
            ("kitchen table", "kitchentable"),
            ("bathroom cabinet", "bathroomcabinet"),
            ("condiment bottle", "condimentbottle"),
            ("remote control", "remotecontrol"),
            ("water glass", "waterglass"),
            ("wine glass", "wineglass"),
            ("bottle of wine", "wine"),
            ("bag of chips", "chips"),
        ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def get_belief(final_vqa, actions):
    for num in [1, 2]:
        goal = final_vqa[f'hypo_goal{num}']

        belief = []
        for possible_belief in POSSIBLE_BELIEF:
            if possible_belief in final_vqa['state']:
                belief.append(possible_belief)

        T = len(actions) - 1
        room = actions[0].split()[-1].strip('.')
        for time in range(1, T+1):
            prev_action = actions[time-1]
            if prev_action == '': continue

            # update person's room
            if prev_action.split()[-1] in ROOM_LIST:
                room = prev_action.split()[-1]
            
            # deal with surfaces in the same room
            sentences = final_vqa['state'].split('.')
            surfaces = list(set(belief) & set(SURFACE_LIST))
            for surface in surfaces:
                surface_inside_room = any(surface in sentence and room in sentence for sentence in sentences)
                if surface_inside_room:
                    goal_on_surface = any(goal in sentence and surface in sentence for sentence in sentences)
                    if goal_on_surface:
                        belief = [surface]
                    else:
                        belief.remove(surface)
                    break

            # deal with opened containers
            if 'open' in prev_action:
                container = ' '.join(prev_action.split()[1:])
                if container in belief:
                    sentences = final_vqa['state'].split('.')
                    goal_inside_container = any(goal in sentence and container in sentence for sentence in sentences)
                    if goal_inside_container and container != 'cabinet':
                        belief = [container]
                    else:
                        belief.remove(container)
            final_vqa[f'goal{num}_belief_{time}'] = ', '.join(belief)
    
    final_vqa = {key: final_vqa.get(key) for key in ["episode", "end_time", "question_type"] + [k for k in final_vqa if k not in ["episode", "end_time", "question_type"]]}

    return final_vqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='multimodal', choices=['multimodal', 'text_only', 'video_only'])
    parser.add_argument("--benchmark_path", type=str, default='questions.json')
    parser.add_argument("--output_file", type=str, default='multimodal_representations.json')
    args = parser.parse_args()

    # Load data
    file_path = args.benchmark_path
    test_vqa_dataset = []
    with open(file_path, "r") as file:
        for line in file:
            test_vqa_dataset.append(json.loads(line))

    file_path = f"inference_raw.json"
    inference_vqa_dataset = []
    with open(file_path, "r") as file:
        for line in file:
            inference_vqa_dataset.append(json.loads(line))


    ## Parse actions
    actions_dataset = []
    for test_vqa in tqdm(test_vqa_dataset):
        test_vqa['question'] = replace_words(test_vqa['question'])

        # Action extraction
        prompt = f"Please extract the exact description about a person's actions (starting from the inital location), found after the phrase '[someone]'s action' and before the question. Please do not include the question, choices, or the answer. \n\
                Input: {test_vqa['question']}. \n\
                Extracted: "
        extracted_actions = generate_response(prompt, max_tokens = 2000).strip()

        # Action parsing
        prompt = f"\
                Please parse the following description about a person's actions. Use a '.' to separate each action, and remove all occurrences of the word 'and' in the description. \n\n\
                Original: Jennifer is in the bedroom. She proceeds to the kitchen and strides towards the oven, preparing to open it. \n\
                Parsed: In the bedroom. walktowards kitchen. walktowards oven. about to open oven. \n\
                Original: Mark is in the bathroom. He then walks to the kitchen. He sequentially approaches the oven, the second, and third kitchen cabinets, opening and closing each one in turn. \n\
                Parsed: In the bedroom. walktowards kitchen. walktowards oven. open oven. close oven. walktowards 2nd kitchencabinet. open 2nd kitchencabinet. close 2nd kitchencabinet. open 3rd kitchencabinet. close 3rd kitchencabinet. \n\
                Original: {extracted_actions} \n\
                Parsed: "
        actions = generate_response(prompt, max_tokens = 2000).strip()
        print("Actions:", actions)
        actions_dataset.append(actions)

    store_parsed_actions = False
    if store_parsed_actions:
        file_path = "actions.json"
        with open(file_path, "w") as file:
            pass
        for actions in actions_dataset:
            with open(file_path, 'a') as file:
                json.dump(actions, file)
                file.write('\n')

    
    ## Parse states
    state_dataset = []
    for test_vqa in tqdm(test_vqa_dataset):
        test_vqa['question'] = replace_words(test_vqa['question'])

        # State extraction
        prompt = f"Please extract the description about the rooms and where things are in an apartment, found after the phrase 'What's inside the apartment' and before the description about a person's actions. Keep the line breaks. \n\
                Question: {test_vqa['question']}. \n\
                Extracted: "
        extracted_state = generate_response(prompt, max_tokens = 2000).strip()

        # State parsing
        prompt = f"\
                Please parse the following description about where things are in an apartment. Each sentence should follow the pattern 'something is/are in/on the location.'  Use a '.' and line break to separate sentences. Keep the original line breaks. \n\n\
                Original: The living room contains a sofa, a desk, a cabinet, and a coffee table, and the cabinet holds chips, a wine glass, and an apple. \n\n\
                Parsed: A sofa, a desk, a cabinet and a coffeetable are in the livingroom. Chips, a wineglass and an apple are in the cabinet. \n\n\
                Original: The kitchen has an oven, a microwave, and four cabinets. The oven contains a salmon, the microwave holds a cupcake, the third cabinet from the left has a wine glass, the fourth cabinet is empty. The first and second kitchen cabinets each holds a plate. \n\n\
                Parsed: an oven and a microwave and 4 kitchencabinets are in the kitchen. A salmon is in the oven. A cupcake is in the microwave. A wineglass is in the 3rd kitchencabinet. Nothing is in the 4th kitchencabinet. A plate is in the 1st kitchencabinet. A plate is in the 2nd kitchencabinet. \n\n\
                Original: {extracted_state} \n\
                Parsed: "
        state = generate_response(prompt, max_tokens = 2000).strip()
        print("State:", state)
        state_dataset.append(state)

    store_parsed_state = False
    if store_parsed_state:
        file_path = "state.json"
        with open(file_path, "w") as file:
            pass
        for state in state_dataset:
            with open(file_path, 'a') as file:
                json.dump(state, file)
                file.write('\n')

    # Load visual perception results
    with open("../closeness.json", "r") as file:
        closeness = json.load(file)


    final_vqa_dataset = []
    line_num = 0
    for i in range(len(test_vqa_dataset)):
        final_vqa = test_vqa_dataset[i]
        final_vqa['question'] = replace_words(final_vqa['question'])

        ## Question parsing
        prompt = f"Please determine the type of inference for the input question: either 'Belief Inference', which inquires about a person's belief regarding an object, or 'Goal Inference', which seeks to understand a person's objective. \n\
                If a question falls under the ``Belief Inference'', please identify the [object] and the [container] that the object may or may not be inside in choices (a) and (b). \n\
                If a question falls under the ``Goal Inference'', please identify the two possible objects that the person is looking for in choices (a) and (b). If the input contains a statement indicating that someone believes there isn't an [object] inside a [container], please also identify both the [object] and the [container] mentioned. Otherwise, return `NaN.' \n\n\
                Input: ... (detailed descriptions about states and actions) ... If Elizabeth has been trying to get a plate, which one of the following statements is more likely to be true? (a) Elizabeth thinks that there is a plate inside the fridge. (b) Elizabeth thinks that there isn't any plate inside the fridge. \n\
                Output: Belief Inference. plate, fridge. \n\n\
                Input: ... (detailed descriptions about states and actions) ... If Jennifer has been trying to get a plate, which one of the following statements is more likely to be true? (a) Jennifer thinks that there is a salmon inside the oven. (b) Jennifer thinks that there isn't any salmon inside the oven. \n\
                Output: Belief Inference. plate, fridge. salmon, oven. \n\n\
                Input: ... (detailed descriptions about states and actions) ... Which one of the following statements is more likely to be true? (a) Mark has been trying to get a plate. (b) Mark has been trying to get a cupcake. \n\
                Output: Goal Inference. plate, cupcake. NaN. \n\n\
                Input: ... (detailed descriptions about states and actions) ... If Mary think there isn't an apple inside the microwave, which one of the following statements is more likely to be true? (a) Mary has been trying to get an apple. (b) Mary has been trying to get a bottle of wine. \n\
                Output: Goal Inference. apple, wine. apple, microwave. \\\\
                Input: {final_vqa['question']} \n\
                Output:"
        
        output = generate_response(prompt, max_tokens = 2000).strip()
        output = output.replace('.', '').replace(',', '').split()
        if output[0] == 'Belief':
            try:
                final_vqa['hypo_goal'], final_vqa['hypo_belief'] = [output.lower() for output in output[2:4]]
            except Exception as e:
                print("Error identifying hypo belief.", final_vqa['question'])
        else:
            try:
                final_vqa['hypo_goal1'], final_vqa['hypo_goal2'] = [output.lower() for output in output[2:4]]
            except Exception as e:
                print("Error identifying hypo goals.", final_vqa['question'])
            try:
                object, container = output[4:6]
                if object == final_vqa['hypo_goal1']:
                    final_vqa['goal1_remove_belief'] = container
                elif object == final_vqa['hypo_goal2']:
                    final_vqa['goal2_remove_belief'] = container
                else:
                    print(f"object not found: {object}, {container}")
            except Exception as e:
                pass


        ## Get parsed actions
        parsed_actions = actions_dataset[i].strip(". ").split(". ")

        # Get init state
        final_vqa['state'] = final_vqa['question'].split("What's inside the apartment: ")[1].split("\nActions")[0]

        if args.type != 'video_only':
            T = final_vqa['end_time'] + 1
            for time in range(T + 1):
                if args.type == 'text_only':
                    final_vqa[f'state_{time}'] = inference_vqa_dataset[i][f'state_{time}']
                else:
                    final_vqa[f'state_{time}'] = final_vqa['state']
                    current_closeness = closeness[str(final_vqa['episode'])][str(time)]
                    for item in current_closeness:
                        final_vqa[f'state_{time}'] += f'The person is close to the {item}. '
                    
            # Get final actions
            actions = []
            actions.append(parsed_actions[0])
            count = 1
            for time in range(1, T+1):
                actions.append(parsed_actions[count])
                noun = parsed_actions[count].split()[-1]
                if 'walk' in parsed_actions[count] and noun in POSSIBLE_BELIEF and time != T and f"close to the {noun}" not in final_vqa[f'state_{time+1}']:
                    continue

                if count < len(parsed_actions) - 1:
                    count += 1
            
            actions[-1] = parsed_actions[-1]
            final_vqa['actions'] = actions

            ## Get belief
            final_vqa = get_belief(final_vqa, actions)
    
        else:
            ## Get states
            containers = []
            for container in POSSIBLE_CONTAINER:
                if container in final_vqa['state']:
                    containers.append(container)

            T = len(parsed_actions) - 1
            for time in range(T + 1):
                final_vqa[f'state_{time}'] = final_vqa['state']
                for container in containers:
                    if 'close '+ container == parsed_actions[time]:
                        final_vqa[f'state_{time}'] += f' The {container} is open.'
                        final_vqa[f'state_{time}'] += f' The person is close to the {container}.'
                    else:
                        final_vqa[f'state_{time}'] += f' The {container} is closed.'
                    if 'open '+ container == parsed_actions[time]:
                        final_vqa[f'state_{time}'] += f' The person is close to the {container}.'
            
            # Get actions
            final_vqa['actions'] = parsed_actions

            ## Get belief
            final_vqa = get_belief(final_vqa, parsed_actions)
        
        final_vqa_dataset.append(final_vqa)


    # Store parsed and processed dataset
    # if args.type == 'multimodal':
    #     file_path = "inference_multimodal.json"
    # elif args.type == 'text_only':
    #     file_path = "inference_text.json"
    # else:
    #     file_path = "inference_video.json"
        
    file_path = args.output_file
    with open(file_path, "w") as file:
        pass
    for vqa in final_vqa_dataset:
        with open(file_path, 'a') as file:
            json.dump(vqa, file)
            file.write('\n')