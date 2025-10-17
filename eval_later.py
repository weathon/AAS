from datasets import load_dataset
dataset = load_dataset("weathon/aas_benchmark", split="train")

from hpsv3 import HPSv3RewardInferencer

inferencer = HPSv3RewardInferencer(device='cuda:2')

import torch
def hpsv3_reward(sample): 
    images_part = [sample["image_original"], sample["image_original"], sample["image_distorted"],  sample["image_distorted"]]
    prompts_part = [
        sample["prompt_original"],
        sample["prompt_distorted"],
        sample["prompt_original"],
        sample["prompt_distorted"]
    ] 
    with torch.no_grad(): 
        with torch.cuda.amp.autocast():
            rewards = inferencer.reward(prompts=prompts_part, image_paths=images_part)
    results = {
        "hpsv3_oiop": rewards[0], # original image, original prompt
        "hpsv3_oidp": rewards[1], # original image, distorted prompt
        "hpsv3_diop": rewards[2], # distorted image, original prompt
        "hpsv3_didp": rewards[3], # distorted image, distorted prompt 
    }
    return results 
  
rewards = []
import tqdm
for sample in tqdm.tqdm(dataset):
    reward = hpsv3_reward(sample)
    rewards.append(reward)

with open("hpsv3_rewards.pkl", "wb") as f:
    import pickle
    pickle.dump(rewards, f)

# hpsv3_reward(dataset[0])   
# dataset = dataset.map(hpsv3_reward)
# dataset.push_to_hub("weathon/aas_benchmark", private=True)