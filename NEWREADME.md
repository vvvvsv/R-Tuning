pararel openlm-research/open_llama_3b：

原始结果：

```
python evaluate.py --model openlm-research/open_llama_3b --domain ID --result pararel_llama_3b_origin
```

prompt:

```
python evaluate_prompt.py --model openlm-research/open_llama_3b --domain ID --result pararel_llama_3b_prompt
```


unsure：

```
python run_pararel.py --model openlm-research/open_llama_3b --method unsure
bash ./scripts/run_finetune_with_lora.sh --model_name_or_path openlm-research/open_llama_3b --dataset_path ../training/training_data/pararel_unsure/ --output_lora_path output_models/finetuned_pararel_llama_3b_unsure_lora
bash ./scripts/run_merge_lora.sh --model_name_or_path openlm-research/open_llama_3b --lora_model_path output_models/finetuned_pararel_llama_3b_unsure_lora --output_model_path output_models/finetuned_pararel_llama_3b_unsure_lora_merged
python evaluate.py --model ~/code/R-Tuning-wsy/LMFlow/output_models/finetuned_pararel_llama_3b_unsure_lora_merged --domain ID --result pararel_llama_3b_unsure
```

unsure_noinfo：

```
python run_pararel.py --model openlm-research/open_llama_3b --method unsure
bash ./scripts/run_finetune_with_lora.sh --model_name_or_path openlm-research/open_llama_3b --dataset_path ../training/training_data/pararel_unsure_noinfo/ --output_lora_path output_models/finetuned_pararel_llama_3b_unsure_noinfo_lora
bash ./scripts/run_merge_lora.sh --model_name_or_path openlm-research/open_llama_3b --lora_model_path output_models/finetuned_pararel_llama_3b_unsure_noinfo_lora --output_model_path output_models/finetuned_pararel_llama_3b_unsure_noinfo_lora_merged
python evaluate.py --model ~/code/R-Tuning-wsy/LMFlow/output_models/finetuned_pararel_llama_3b_unsure_noinfo_lora_merged --domain ID --result pararel_llama_3b_unsure_noinfo
```

unknown：

```
python run_pararel.py --model openlm-research/open_llama_3b --method unknown
bash ./scripts/run_finetune_with_lora.sh --model_name_or_path openlm-research/open_llama_3b --dataset_path ../training/training_data/pararel_unknown/ --output_lora_path output_models/finetuned_pararel_llama_3b_unknown_lora
bash ./scripts/run_merge_lora.sh --model_name_or_path openlm-research/open_llama_3b --lora_model_path output_models/finetuned_pararel_llama_3b_unknown_lora --output_model_path output_models/finetuned_pararel_llama_3b_unknown_lora_merged
python evaluate_unknown.py --model ~/code/R-Tuning-wsy/LMFlow/output_models/finetuned_pararel_llama_3b_unknown_lora_merged --domain ID --result pararel_llama_3b_unknown
```

DPO：

```
python run_pararel.py --model openlm-research/open_llama_3b --method unknown
bash ./scripts/run_finetune_with_lora.sh --model_name_or_path openlm-research/open_llama_3b --dataset_path ../training/training_data/pararel_dpo/ --output_lora_path output_models/finetuned_pararel_llama_3b_dpo_lora
bash ./scripts/run_merge_lora.sh --model_name_or_path openlm-research/open_llama_3b --lora_model_path output_models/finetuned_pararel_llama_3b_dpo_lora --output_model_path output_models/finetuned_pararel_llama_3b_dpo_lora_merged
python evaluate_dpo.py --model ~/code/R-Tuning-wsy/LMFlow/output_models/finetuned_pararel_llama_3b_dpo_lora_merged --domain ID --result pararel_llama_3b_dpo
```