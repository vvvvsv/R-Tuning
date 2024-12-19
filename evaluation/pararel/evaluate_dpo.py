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

def inference(input_text):

    full_input = f"Question: {input_text} High Quality Answer:"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 10,
                output_scores = True,
                return_dict_in_generate=True
            )
    logits = outputs['scores']
    output_sequence = []
    product = torch.tensor(1.0, device='cuda:0')
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

    dot_index = output_text.find('.')
    output_text = output_text[:dot_index+1] if dot_index != -1 else output_text

    return output_text, full_input, np.power(product.item(),(1/count)).item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="pararel")
    parser.add_argument('--domain',type=str, default="ID")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.'
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    data = []
    with open(f"../../dataset/pararel/{args.domain}_test_pararel.json",'r') as f:
        data = json.load(f)

    results = []
    results_categories = {}
    global_total = 0
    global_correct = 0
    # sample[0] is question. sample[1] is answer.
    for sample in tqdm(data):
        output, full_input, predict_conf = inference(sample[0])
        # sure_prob > 0.5 -> I am sure. Otherwise I am unsure
        if sample[2] not in results_categories:
            results_categories[sample[2]] = {"total": 0, "correct": 0}
        global_total += 1
        results_categories[sample[2]]['total'] += 1

        result = 0 # 0 denotes wrong prediction
        if sample[1] in output:
            global_correct += 1
            results_categories[sample[2]]['correct'] += 1
            result = 1 # 1 denotes correct prediction
        else:
            for ans in UNKNOWN_RESPONSES:
                if ans in output:
                    result = -1 # -1 denotes refusal
                    break

        results.append((result, sample, output))

        torch.cuda.empty_cache()

    for (category, data) in results_categories.items():
        print("Category: {}, Acc: {:.2%}, Total: {}, Correct: {}"
              .format(category,
                      data['correct']/data['total'],
                      data['total'],
                      data['correct']))
    print("Global Acc: {:.2%}, Total: {}, Correct: {}"
          .format(global_correct/global_total, global_total, global_correct))

    os.makedirs("results",exist_ok=True)
    with open(f"results/{args.result}_{args.domain}.json",'w') as f:
        json.dump(results,f)

