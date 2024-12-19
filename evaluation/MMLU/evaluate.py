from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
import warnings

STOP = []
SURE = []
UNSURE = []

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "\nAnswer:"
    return prompt

def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt(input_list,subject,prompt_data):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_shots(prompt_data)
    prompt += format_example(input_list)
    return prompt

def inference(tokenizer,model,input_text,subject,prompt_data):
    full_input = gen_prompt(input_text,subject,prompt_data)
    inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True).to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )
    logits = outputs['scores'][0][0]    #The first token
    probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
    )
    output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    conf = np.max(probs)

    return output_text, full_input, conf.item()

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input,return_tensors="pt", padding=True, truncation=True).to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True,
                attention_mask=inputs["attention_mask"],
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )
    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization

    return sure_prob.item()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')
    model.config.pad_token_id = tokenizer.pad_token_id

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    results = []
    data = {}
    prompt = {}
    with open(f"../../dataset/MMLU/MMLU_{args.domain}_test.json",'r') as f:
        data = json.load(f)

    with open(f"../../dataset/MMLU/MMLU_{args.domain}_prompt.json",'r') as f:
        prompt = json.load(f)

    acc_dict = {}

    for i in tqdm(data.keys()):
        prompt_data = prompt[i]
        type_name = i
        sum_right = 0
        total_sure_prob = 0
        correct_sure_prob = 0

        for instance in tqdm(data[i]):
            output,full_input, predict_conf = inference(tokenizer,model,instance,i,prompt_data)
            sure_prob = checksure(f"{full_input}{output}")
            if instance[-1] in output:
                results.append((1,predict_conf,sure_prob))   # 1 denotes correct prediction
                sum_right += 1
                correct_sure_prob += sure_prob
            else:
                results.append((0,predict_conf,sure_prob))   # 0 denotes wrong prediction
            total_sure_prob += sure_prob

        num_instances = len(data[i])
        accuracy = sum_right / num_instances
        print(f"Finished evaluation for {i}, accuracy: {accuracy:.2%}")
        acc_dict[i] = (
            f"Accuracy: {accuracy:.2%}, "
        )

        torch.cuda.empty_cache()

    total_instances = sum(len(data[i]) for i in data.keys())
    total_accuracy = sum(r[0] for r in results) / total_instances
    total_avg_sure_prob = sum(r[2] for r in results) / total_instances
    total_avg_correct_sure_prob = sum(r[2] for r in results if r[0] == 1) / sum(r[0] for r in results)

    # 打印总的五个指标
    print("\nOverall Results:")
    print(f"Total Accuracy: {total_accuracy:.2%}")
    print(f"Total Avg Sure Probability: {total_avg_sure_prob:.2%}")
    print(f"Total Avg Correct Sure Probability: {total_avg_correct_sure_prob:.2%}")
    print(acc_dict)

    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}_{args.domain}.json",'w') as f:
        json.dump(results,f)
