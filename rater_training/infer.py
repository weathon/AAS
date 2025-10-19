import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "weathon/smolvlm2_anti_aesthetics_7b", dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

from datasets import load_dataset

import pandas as pd

df = pd.read_csv("rules.csv")
sign = lambda x: (x > 0) - (x < 0)
import pandas as pd
import re

df.columns = df.columns.str.strip()
df['Dimension'] = df['Dimension'].ffill()

df['dim_key'] = df['Dimension'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1) if re.search(r'\((.*?)\)', x) else x)

guide = {
    dim_key: {
        f"{row['Option']}: {sign(int(row['Score']))+1}": ": " +str(row['Description']).strip()
        for _, row in group.iterrows()
    }
    for dim_key, group in df.groupby('dim_key')
}

score = {
    dim_key: {
        int(row['Score']): str(row['Option']).strip()
        for _, row in group.iterrows()
    }
    for dim_key, group in df.groupby('dim_key')
}



import json
import torch

def rate_single_image(image):
    results = {"scores": [], "preds": []}
    messages = []
    for dim in guide.keys():
        messages.append([
                    { 
                        "role": "user",
                        "content": [
                            {
                                "type":"text",
                                "text":f"Please rate this image for its {dim} quality. Use this guideline {guide[dim]}. Response a single number.",
                            },
                            {
                                "type": "image", 
                                "image":image.resize((512, 512))
                            }
                        ],
                    }])

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ) 

    inputs = inputs.to(model.device)
    id_of_interest = processor.tokenizer.convert_tokens_to_ids(["0", "1", "2"])
    id_of_interest
    with torch.no_grad():
        logits = model(**inputs, max_new_tokens=128).logits
    logits = logits[:, -1, id_of_interest]
    prob = torch.softmax(logits, dim=-1)
    for i in range(len(prob)):
        prob_of_interest = prob[i]
        score = torch.dot(prob_of_interest, torch.tensor([0, 1, 2], device=prob_of_interest.device).bfloat16())
        single_pred = prob_of_interest.argmax().item()
        # results[f"{dim}_score"] = float(score)
        # results[f"{dim}_pred"] = single_pred
        print(list(guide.keys())[i], float(score))
        results["scores"].append(float(score))
        results["preds"].append(single_pred)
    return results

dataset = load_dataset("weathon/aas_benchmark")
dataset["train"] = dataset["train"].remove_columns(["hpsv2"])
dataset = dataset["train"]

def rate_image(sample):
    from PIL import Image
    image_original = sample["image_original"]
    results_original = rate_single_image(image_original)
    image_distorted = sample["image_distorted"]
    results_distorted = rate_single_image(image_distorted)
    # sample["rater"] = 
    return {
        "original": results_original,
        "distorted": results_distorted
    }
    # return sample
import tqdm
# https://huggingface.co/docs/datasets/en/process
if __name__ == "__main__":
    # rated_dataset = dataset.map(rate_image, batched=False, writer_batch_size=3000)
    rater_results = []
    for sample in tqdm.tqdm(dataset):
        result = rate_image(sample)
        rater_results.append(result)
    dataset = dataset.add_column("rater", rater_results)
    dataset.push_to_hub("weathon/aas_benchmark")

