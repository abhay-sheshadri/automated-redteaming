from online_attacks import PromptAutomaticIterativeRefinement
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from strongreject_evaluator import autograde_response


load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")
openai_api_key = "YOUR API KEY"
os.environ["OPENAI_API_KEY"] = openai_api_key

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_dtype = torch.bfloat16
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_access_token,
    torch_dtype=model_dtype
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

dataset = load_dataset("abhayesian/augmented_advbench_v5")["train"].shuffle()

pair = PromptAutomaticIterativeRefinement(
    model=model,
    tokenizer=tokenizer,
    api_key=openai_api_key,
)

with open("pair_jailbreaks.json", "r") as f:
    results = json.load(f)

for i in range(len(dataset)):
    try:
        prompt = dataset[i]["prompt"]
        target = " ".join(dataset[i]["rejected"].split()[:10])
        chosen_response = dataset[i]["chosen"]
        
        # Get PAIR attack
        top_candidate, high_score, response = pair.compute_adversary(prompt, target)

        # Get StrongReject response
        score = autograde_response(prompt, response).score_zero_to_one
        
        if score == 0:
            continue
        
        results.append({
            "prompt": prompt,
            "refusal": chosen_response,
            "jailbreak": top_candidate,
            "model_response": response,
            "pair_score": high_score,
            "strongreject_score": score
        })
        
        with open("pair_jailbreaks.json", "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        print(e)
