import io
import time
import json
import base64
import pickle
import random
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
random.seed(42)
# openai.api_key = YOUR-OPENAI-KEY

def generate_response(prompt, max_tokens = 100, temperature = 0):
    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response["choices"][0]["text"]

def generate_chat_response(prompt, max_tokens = 100, temperature = 0):
    response = openai.ChatCompletion.create(
        model = "gpt-4",    # choose in "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response.choices[0].message.content

def generate_response_gpt4v(prompt, base64_images, max_tokens = 100, temperature = 0):
    client = OpenAI(api_key='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')
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[1]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[2]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[3]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[4]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[5]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[6]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[7]}",
                    "detail": "low"
                    },
                },
            ],
            }
        ],
        max_tokens = max_tokens,
        )
    return response.choices[0].message.content

def read_frame_intervals(parent_path):
    path = parent_path + 'frame_intervals.pik'
    with open(path, 'rb') as f:
        intervals = pickle.load(f)
    return intervals

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        # Create an empty image and return its encoding if the file is not found
        empty_image = Image.new('RGB', (1, 1))
        buffered = io.BytesIO()
        empty_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", type=str, default='questions.json')
    parser.add_argument("--benchmark_video_path", type=str, default='videos')
    parser.add_argument("--gpt_choice", type=str, default='gpt-4v', choices=['gpt-3.5', 'gpt-4', 'gpt-4v'])
    parser.add_argument("--few_shot", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--video_only", action='store_true')
    args = parser.parse_args()

    # Load data
    file_path = args.benchmark_path
    vqa_dataset = []
    with open(file_path, "r") as file:
        for line in file:
            vqa_dataset.append(json.loads(line))

    # Few shot prompts
    two_shot_prompt = "What's inside the apartment: The apartment consists of a living room, bedroom, kitchen, and bathroom. \nIn the living room, there is a coffee table, a sofa, a cabinet, and a desk. The cabinet houses a dish bowl, a water glass, a bag of chips, and a bottle of wine. \nThe bedroom is furnished with a desk and a coffee table, which has three water glasses on it. \nThe kitchen is equipped with four cabinets, a fridge, a microwave, a kitchen table, and a stove. The first cabinet, from left to right, contains a plate and a bottle of wine. The second cabinet holds a wine glass and a water glass. The third cabinet stores two apples, while the fourth cabinet has a wine glass and an apple. The fridge contains a salmon and an apple. Inside the microwave, there is a condiment bottle and a salmon. The stove houses a cupcake and a plate. \nThe bathroom features a cabinet, which is currently empty. \nActions taken by Mary: Mary is situated in the living room. She proceeds to the kitchen and heads towards the second kitchen cabinet. She opens it, then promptly closes it. Mary then makes her way to the bathroom, approaching the bathroom cabinet. She opens it, then shuts it. She returns to the kitchen, moving towards the microwave, which she opens and then closes. She then opens the fridge, and after a moment, closes it. Mary then walks towards the third kitchen cabinet, opens it, and closes it. She repeats this action with the fourth kitchen cabinet. She then opens the stove, and after a moment, closes it. She walks towards the first kitchen cabinet, opens it, and then closes it. Finally, Mary returns to the living room, preparing to open another cabinet. \nQuestion: If Mary has been trying to get a bag of chips, which one of the following statements is more likely to be true? (a) Mary thinks that the bag of chips is not inside the cabinet. (b) Mary thinks that the bag of chips is inside the cabinet. Please respond with either a or b. \nAnswer: b. \n\n\n\
                    What's inside the apartment: The apartment consists of a bedroom, bathroom, living room, and kitchen. \nIn the bedroom, there is a desk and a coffee table, with a dish bowl placed on the coffee table. \nThe bathroom houses a cabinet, which is currently empty. \nThe living room is furnished with a desk, coffee table, cabinet, and sofa. On the coffee table, there is a remote control, a wine glass, and a dish bowl. The cabinet contains a bag of chips, a water glass, a condiment bottle, and two wine glasses. \nThe kitchen is equipped with four cabinets, a fridge, a stove, a microwave, and a kitchen table. The first kitchen cabinet, from left to right, contains an apple, while the fourth is empty. Inside the fridge, there are two cupcakes, a bottle of wine, three apples, a dish bowl, and a plate. The third kitchen cabinet is empty. The stove houses a salmon. The second kitchen cabinet contains a dish bowl and a wine glass. Lastly, there is a cupcake in the microwave. \nActions taken by Elizabeth: Elizabeth is situated in the kitchen. She strides towards the first kitchen cabinet, opens it, then promptly shuts it. Subsequently, she opens the third kitchen cabinet and closes it as well, before finally making her way towards the fridge. \nQuestion: If Elizabeth doesn't think there is a bottle of wine inside the fridge, which one of the following statements is more likely to be true? (a) Elizabeth has been trying to get a remote control. (b) Elizabeth has been trying to get a bottle of wine. Please respond with either a or b. \nAnswer: a. \n\n\n"
    one_shot_prompt = "What's inside the apartment: The apartment consists of a living room, bedroom, kitchen, and bathroom. \nIn the living room, there is a coffee table, a sofa, a cabinet, and a desk. The cabinet houses a dish bowl, a water glass, a bag of chips, and a bottle of wine. \nThe bedroom is furnished with a desk and a coffee table, which has three water glasses on it. \nThe kitchen is equipped with four cabinets, a fridge, a microwave, a kitchen table, and a stove. The first cabinet, from left to right, contains a plate and a bottle of wine. The second cabinet holds a wine glass and a water glass. The third cabinet stores two apples, while the fourth cabinet has a wine glass and an apple. The fridge contains a salmon and an apple. Inside the microwave, there is a condiment bottle and a salmon. The stove houses a cupcake and a plate. \nThe bathroom features a cabinet, which is currently empty. \nActions taken by Mary: Mary is situated in the living room. She proceeds to the kitchen and heads towards the second kitchen cabinet. She opens it, then promptly closes it. Mary then makes her way to the bathroom, approaching the bathroom cabinet. She opens it, then shuts it. She returns to the kitchen, moving towards the microwave, which she opens and then closes. She then opens the fridge, and after a moment, closes it. Mary then walks towards the third kitchen cabinet, opens it, and closes it. She repeats this action with the fourth kitchen cabinet. She then opens the stove, and after a moment, closes it. She walks towards the first kitchen cabinet, opens it, and then closes it. Finally, Mary returns to the living room, preparing to open another cabinet. \nQuestion: If Mary has been trying to get a bag of chips, which one of the following statements is more likely to be true? (a) Mary thinks that the bag of chips is not inside the cabinet. (b) Mary thinks that the bag of chips is inside the cabinet. Please respond with either a or b. \nAnswer: b. \n\n\n"

    correct = {}
    all = {}
    answers = []
    for vqa in tqdm(vqa_dataset):
        if args.gpt_choice == 'gpt-4v':
            episode = vqa["episode"]
            end_time = vqa["end_time"]
            interval_path = f'{args.benchmark_video_path}/task_{episode}/'
            video_path = f'{args.benchmark_video_path}/task_{episode}/script/0/'

            # Read frame intervals and extracting end frame time
            intervals = read_frame_intervals(interval_path)
            times = [action[1] for action in intervals]
            end_frame = times[end_time]

            # Construct paths for selected frames and encoding them
            num_frame = 8
            step_size = int(end_frame / (num_frame - 1))
            selected_numbers = [i * step_size for i in range(num_frame)]
            paths = [video_path + f'Action_{selected_number:04d}_0_normal.png' for selected_number in selected_numbers]
            base64_images = [encode_image(path) for path in paths]

        # Construct the question prompt based on the modality and few-shot setting
        question = vqa["question"]
        if args.video_only:
            question = "\nQuestion:" + question.split("\nQuestion:")[-1]
        if args.few_shot == 2:
            question_prompt = two_shot_prompt + question + "\nAnswer: " # + "Let's think step by step."
        elif args.few_shot == 1:
            question_prompt = one_shot_prompt + question + "\nAnswer: " # + "Let's think step by step."
        else:
            question_prompt = question + "\nAnswer: " # + "Let's think step by step."

        # Generate an answer
        if args.gpt_choice == 'gpt-4':
            generated_answer = generate_chat_response(question_prompt, max_tokens = 5).strip("\n").strip(" ").lower()[0]
            if args.two_shot:
                time.sleep(2)
        elif args.gpt_choice == 'gpt-3.5':
            generated_answer = generate_response(question_prompt, max_tokens = 5).strip("\n").strip(" ").lower()[0]
        else:
            generated_answer = generate_response_gpt4v(question_prompt, base64_images, max_tokens = 5).strip("\n").strip(" ").lower()[0] 

        # Get the correct answer
        correct_answer = vqa['answer']

        VERBOSE = True
        if VERBOSE:
            print(question_prompt)
            print(vqa['test'])
            print(vqa['question_type'])
            print("\n")
            print("Generate answer: " + generated_answer)
            print("\n")
            print("Correct answer: " + correct_answer)
            print("\n")

        # Evaluate the correctness
        if generated_answer == correct_answer:
            correct[f"{str(vqa['test'])} + type {vqa['question_type']}"] = correct.get(f"{str(vqa['test'])} + type {vqa['question_type']}", 0) + 1
            answers.append(1)
        else:
            answers.append(0)
        all[f"{str(vqa['test'])} + type {vqa['question_type']}"] = all.get(f"{str(vqa['test'])} + type {vqa['question_type']}", 0) + 1


    # Show results
    tests = [['filtered_graph', 'filtered_actions'], ['filtered_init_graph', 'actions'], ['graph', 'filtered_actions'], ['init_graph', 'actions']]
    question_types = [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]
    for test in tests:
        for question_type in question_types:
            index = f"{str(test)} + type {question_type}"
            print(f"{index}: {correct.get(index, 0)} out of {all.get(index, 0)} questions are answered correctly.")
    
    # Create a table to show results
    results = {
        '': [f"type {question_type}" for question_type in question_types],
        str(tests[0]): [f"{correct.get(f'{str(tests[0])} + type {question_type}', 0)}/{all.get(f'{str(tests[0])} + type {question_type}', 0)}" for question_type in question_types],
        str(tests[1]): [f"{correct.get(f'{str(tests[1])} + type {question_type}', 0)}/{all.get(f'{str(tests[1])} + type {question_type}', 0)}" for question_type in question_types],
        str(tests[2]): [f"{correct.get(f'{str(tests[2])} + type {question_type}', 0)}/{all.get(f'{str(tests[2])} + type {question_type}', 0)}" for question_type in question_types],
        str(tests[3]): [f"{correct.get(f'{str(tests[3])} + type {question_type}', 0)}/{all.get(f'{str(tests[3])} + type {question_type}', 0)}" for question_type in question_types],
    }
    df = pd.DataFrame(results)
    print(df)
    print(answers)