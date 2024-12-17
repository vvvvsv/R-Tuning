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

STOP = []
SURE = []
UNSURE = []

choices = ["A", "B", "C", "D"]

UNKNOWN_RESPONSES = [
    "unknown",
    "not known",
    "n't known",
    "uncertain",
    "not certain",
    "n't certain",
    "unclear",
    "not clear",
    "n't clear",
    "unsure",
    "not sure",
    "n't sure",
    "do not know",
    "no definitive",
    "debate",
    "no concrete",
    "impossible",
    "no known",
    "no public information available",
    "no scientific evidence",
    "no right answer",
    "impossible",
    "difficult",
]

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
        prompt += data[k+1] + ".\n\n"

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
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    outputs = model.generate(
                ids,
                max_new_tokens = 20,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
    output_sequence = []
    product = 1
    count = 0
    for i in logits:        #greedy decoding and calculate the confidence
        pt = torch.softmax(torch.Tensor(i[0]),dim=0)
        max_loc = torch.argmax(pt)

        if max_loc in STOP:
            break
        else:
            output_sequence.append(max_loc)
            product *= torch.max(pt)
            count += 1

    if output_sequence:
        output_text = tokenizer.decode(output_sequence)
    else:
        output_text = ""

    return output_text, full_input, conf.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    results = []
    data = {}
    prompt = {}
    with open(f"../../dataset/MMLU/{args.dataset}.json",'r') as f:
        data = json.load(f)

    with open(f"../../dataset/MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)

    for i in tqdm(data.keys()):
        prompt_data = prompt[i]
        type_name = i
        for instance in tqdm(data[i]):
            output,full_input = inference(tokenizer,model,instance,i,prompt_data)

            result = 0 # 0 denotes wrong prediction

            if instance[1] in output:
                result = 1 # 1 denotes correct prediction
            else:
                for ans in UNKNOWN_RESPONSES:
                    if ans in output:
                        result = -1 # -1 denotes refusal
                        break
            results.append((result, full_input, output))

        torch.cuda.empty_cache()

    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}_{args.domain}.json",'w') as f:
        json.dump(results,f)

