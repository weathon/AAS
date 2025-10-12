# %%
import random
from ollama import chat
from pydantic import BaseModel

class Prompts(BaseModel):
    long_description: str
    short_description: str

# def get_prompt(original_prompt, artifacts):
#     while True:
#         try:
#             messages = [
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f'Write an image description based on {original_prompt}. The picture has effects of {artifacts}. \
#             Specifically, these effects are prioritized over the original subject. Make the effects concrete, for example if the attribute says dark \
#             describe it as a stormy night, and describe the grainy attribute as rough, digitized film grain. But still keep the raw attribute in it, (i.e., still mention plain "dark", "blurry", etc) You should provide two responses, one long one and the \
#             other one has the entire description must be under 50 words and \
#             contain only the image statement. (i.e. no "here it is", "this is the description", etc.) /no_think'},
#             ]
#             response = chat('qwen3:30b-a3b', messages=messages, options={"temperature":0.001}, format=Prompts.model_json_schema())
#             return Prompts.model_validate_json(response["message"]['content'])
#         except Exception as e:
#             print("Error occurred, retrying...", e)
#             continue

import random
from openai import OpenAI
from pydantic import BaseModel

# client = OpenAI(
#     base_url="http://127.0.0.1:8000/v1",
#     api_key="token-abc123"
# )

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

def get_prompt(original_prompt, artifacts):
    while True:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f'Write an image description based on {original_prompt}. The picture has effects of {artifacts}. '
                                            'Specifically, these effects are prioritized over the original subject. Make the effects concrete, like describing specificly what makes the image look that way.'
                                            'You should provide two responses, one long (less than 100 words, could be less if needed) one and the '
                                            'other one has the entire description must be under 50 words'
                                            'You should **only** apply the effects you are given, do not add other effects to couple with it. Do not add details that are not related to the effects or original prompt. '},
            ]
            response = client.chat.completions.parse(
                model="qwen/qwen3-235b-a22b-2507",
                messages=messages,
                response_format=Prompts,
                temperature=0.1
            )
            return response.choices[0].message.parsed
        except Exception as e:
            print("Error occurred, retrying...", e, artifacts)
            continue


# %%
from datasets import load_dataset
coco = load_dataset("raniatze/coco_stuff_train2017_captioned", split="train[0:1000]")

# %%
import pandas as pd
import re
from PIL import Image
import random
df = pd.read_csv("gen_rules.csv")
df.columns = df.columns.str.strip()
df['Dimension'] = df['Dimension'].ffill()

df['dim_key'] = df['Dimension'].apply(lambda x: re.search(r'\((.*?)\)', x).group(1) if re.search(r'\((.*?)\)', x) else x)

guide = {
    dim_key: {
        int(row['Score']): str(row['Description']).strip()
        for _, row in group.iterrows()
    }
    for dim_key, group in df.groupby('dim_key')
}

# %%
negative_prompts = {
    "symmetry": "symmetrical, high symmetry",
    "object pairing": "serenity, dynamism, harmony, resulting, overall coordination, visual unity, complementary relationships",
    "main object": "big noticeable main object",
    "richness": "many objects and small details, visually full or detailed",
    "background": "beautiful background",
    "clarity": "clear, sharpen, clarify",
    "color brightness": "bright color",
    "color aesthetic": "beautiful, nature, normal colors",
    "lighting distinction": "pronounced lighting, shadows, reflections, refractions",
    "lighting aesthetic": "pronounced lighting, shadows, reflections, refractions.",
    "emotion": "happy, joyful, cheerful, warmth, positive emotions",
    "detail refinement": "refined details",
    "detail realism": "photorealistic, authentic"
}

# %%

import json
with open("prompts.json", "r") as f:
    prompt_dict = json.load(f)

def re_prompt(sample):
  original_prompt = sample["text"]
  applied_keys = random.sample(list(guide.keys()), k=random.randint(1, 4))
  artifacts = [guide[key] for key in applied_keys]
  desc = []
  for key in applied_keys:
    desc.append(prompt_dict[key])

  selected = applied_keys
  
    
  desc = "\n".join(desc)
  prompt = get_prompt(original_prompt, desc)
  sample["disorted_long_prompt"] = prompt.long_description
  sample["disorted_short_prompt"] = prompt.short_description
  sample["selected"] = selected
  sample["desc"] = desc
  sample["negative_prompt"] = ", ".join(negative_prompts[i] for i in selected)

  return {
      "original_prompt": original_prompt,
      "disorted_long_prompt": sample["disorted_long_prompt"],
      "disorted_short_prompt": sample["disorted_short_prompt"],
      "selected": sample["selected"],
      "desc": sample["desc"],
      "negative_prompt": sample["negative_prompt"],
  }

# %% 
re_prompt(coco[0])

# %%
dataset = coco.map(re_prompt, num_proc=10)

# %%
dataset = dataset.remove_columns(["text", "image", "conditioning_image"])

dataset.push_to_hub("weathon/anti_aesthetics_dataset")


