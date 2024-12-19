import json

WITHIN_KNOWLEDGE_BOUNDARY = set()

def parse_origin():
    with open('pararel_llama_3b_origin_ID.json', 'r') as file:
        origin_data = json.load(file)

    total = 0
    correct = 0
    wrong = 0
    categories_dict = {}
    for (result, predict_conf, sure_prob, sample, output) in origin_data:
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            correct += 1
            categories_dict[sample[2]]['correct'] += 1
            WITHIN_KNOWLEDGE_BOUNDARY.add(sample[0])
        elif result == 0:
            wrong += 1
            categories_dict[sample[2]]['wrong'] += 1

    print("## Origin:\nCorrect: {}, Wrong: {}\nAccuracy: {:.2%}, Total: {}\n"
          .format(correct, wrong, correct / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, wrong, correct / total, total))
    print()

def parse_prompt():
    with open('pararel_llama_3b_prompt_ID.json', 'r') as file:
        prompt_data = json.load(file)

    total = 0
    correct = 0
    over_refuse = 0
    refuse = 0
    wrong = 0
    categories_dict = {}
    for (result, sample, output) in prompt_data:
        if 't know' in output:
            result = -1
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'over_refuse': 0, 'refuse': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            correct += 1
            categories_dict[sample[2]]['correct'] += 1
        elif result == 0:
            wrong += 1
            categories_dict[sample[2]]['wrong'] += 1
        elif result == -1:
            if sample[0] in WITHIN_KNOWLEDGE_BOUNDARY:
                over_refuse += 1
                categories_dict[sample[2]]['over_refuse'] += 1
            else:
                refuse += 1
                categories_dict[sample[2]]['refuse'] += 1

    print("## Prompt:\nCorrect: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}\nAccuracy: {:.2%}, Expected: {:.2%}, Total: {}\n"
          .format(correct, over_refuse, refuse, wrong, correct / total, (correct + refuse) / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        over_refuse = categories_dict[category]['over_refuse']
        refuse = categories_dict[category]['refuse']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, over_refuse, refuse, wrong, correct / total, total))
    print()


UNSURE_THREHOLD = 0.5
def parse_unsure():
    with open('pararel_llama_3b_unsure_ID.json', 'r') as file:
        unsure_data = json.load(file)

    total = 0
    correct = 0
    over_refuse = 0
    refuse = 0
    wrong = 0
    categories_dict = {}
    for (result, predict_conf, sure_prob, sample, output) in unsure_data:
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'over_refuse': 0, 'refuse': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            if sure_prob > UNSURE_THREHOLD:
                correct += 1
                categories_dict[sample[2]]['correct'] += 1
            else:
                over_refuse += 1
                categories_dict[sample[2]]['over_refuse'] += 1
        elif result == 0:
            if sure_prob > UNSURE_THREHOLD:
                wrong += 1
                categories_dict[sample[2]]['wrong'] += 1
            else:
                refuse += 1
                categories_dict[sample[2]]['refuse'] += 1

    print("## Unsure:\nCorrect: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}\nAccuracy: {:.2%}, Expected: {:.2%}, Total: {}\n"
          .format(correct, over_refuse, refuse, wrong, correct / total, (correct + refuse) / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        over_refuse = categories_dict[category]['over_refuse']
        refuse = categories_dict[category]['refuse']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, over_refuse, refuse, wrong, correct / total, total))
    print()

def parse_unsure_noinfo():
    with open('pararel_llama_3b_unsure_noinfo_ID.json', 'r') as file:
        unsure_noinfo_data = json.load(file)

    total = 0
    correct = 0
    over_refuse = 0
    refuse = 0
    wrong = 0
    categories_dict = {}
    for (result, predict_conf, sure_prob, sample, output) in unsure_noinfo_data:
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'over_refuse': 0, 'refuse': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            if sure_prob > UNSURE_THREHOLD:
                correct += 1
                categories_dict[sample[2]]['correct'] += 1
            else:
                over_refuse += 1
                categories_dict[sample[2]]['over_refuse'] += 1
        elif result == 0:
            if sure_prob > UNSURE_THREHOLD:
                wrong += 1
                categories_dict[sample[2]]['wrong'] += 1
            else:
                refuse += 1
                categories_dict[sample[2]]['refuse'] += 1

    print("## Unsure-NoInfo:\nCorrect: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}\nAccuracy: {:.2%}, Expected: {:.2%}, Total: {}\n"
          .format(correct, over_refuse, refuse, wrong, correct / total, (correct + refuse) / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        over_refuse = categories_dict[category]['over_refuse']
        refuse = categories_dict[category]['refuse']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, over_refuse, refuse, wrong, correct / total, total))
    print()


def parse_unknown():
    with open('pararel_llama_3b_unknown_ID.json', 'r') as file:
        unknown_data = json.load(file)

    total = 0
    correct = 0
    over_refuse = 0
    refuse = 0
    wrong = 0
    categories_dict = {}
    for (result, sample, output) in unknown_data:
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'over_refuse': 0, 'refuse': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            correct += 1
            categories_dict[sample[2]]['correct'] += 1
        elif result == 0:
            wrong += 1
            categories_dict[sample[2]]['wrong'] += 1
        elif result == -1:
            if sample[0] in WITHIN_KNOWLEDGE_BOUNDARY:
                over_refuse += 1
                categories_dict[sample[2]]['over_refuse'] += 1
            else:
                refuse += 1
                categories_dict[sample[2]]['refuse'] += 1

    print("## Unknown:\nCorrect: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}\nAccuracy: {:.2%}, Expected: {:.2%}, Total: {}\n"
          .format(correct, over_refuse, refuse, wrong, correct / total, (correct + refuse) / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        over_refuse = categories_dict[category]['over_refuse']
        refuse = categories_dict[category]['refuse']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, over_refuse, refuse, wrong, correct / total, total))
    print()


def parse_dpo():
    with open('pararel_llama_3b_dpo_ID.json', 'r') as file:
        dpo_data = json.load(file)

    total = 0
    correct = 0
    over_refuse = 0
    refuse = 0
    wrong = 0
    categories_dict = {}
    for (result, sample, output) in dpo_data:
        total += 1

        if sample[2] not in categories_dict:
            categories_dict[sample[2]] = {"total": 0, 'correct': 0, 'over_refuse': 0, 'refuse': 0, 'wrong': 0}
        categories_dict[sample[2]]['total'] += 1

        if result == 1:
            assert(sample[1] in output)
            correct += 1
            categories_dict[sample[2]]['correct'] += 1
        elif result == 0:
            wrong += 1
            categories_dict[sample[2]]['wrong'] += 1
        elif result == -1:
            if sample[0] in WITHIN_KNOWLEDGE_BOUNDARY:
                over_refuse += 1
                categories_dict[sample[2]]['over_refuse'] += 1
            else:
                refuse += 1
                categories_dict[sample[2]]['refuse'] += 1

    print("## DPO:\nCorrect: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}\nAccuracy: {:.2%}, Expected: {:.2%}, Total: {}\n"
          .format(correct, over_refuse, refuse, wrong, correct / total, (correct + refuse) / total, total))

    for category in sorted(list(categories_dict.keys())):
        total = categories_dict[category]['total']
        correct = categories_dict[category]['correct']
        over_refuse = categories_dict[category]['over_refuse']
        refuse = categories_dict[category]['refuse']
        wrong = categories_dict[category]['wrong']

        print("Category: {}, Correct: {}, Over-Refuse: {}, Refuse: {}, Wrong: {}, Accuracy: {:.2%}, Total: {}"
          .format(category, correct, over_refuse, refuse,  wrong, correct / total, total))
    print()


if __name__ == '__main__':
    parse_origin()
    parse_prompt()
    parse_unsure()
    parse_unsure_noinfo()
    parse_unknown()
    parse_dpo()