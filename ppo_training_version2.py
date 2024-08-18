# Imports

from tqdm import tqdm
import os

import torch

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification

from dataset import get_dataset

import json
import numpy as np


tokenizer_path = '/content/drive/MyDrive/NLP/SFT_GPT-2M_Dolly15k'
path = "/content/drive/MyDrive/NLP/Method1_dataset.csv"


# Load the dataset
subreddit_question_dataset = get_dataset(path, tokenizer_path)

# Define a collator needed for the PPO object


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=512)
tokenizer.pad_token = tokenizer.eos_token

# PPO Config
config = PPOConfig(
    model_name='gpt2-medium',
    batch_size=16,
    mini_batch_size=8,
    steps=125,
)

# Load the SFT model, and the required reference model is created in a
# way that it can share layers with the 'current policy' model.

model = AutoModelForCausalLMWithValueHead.from_pretrained(tokenizer_path)
ref_model = create_reference_model(model)
# ref_model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2-medium')

# Required environment setups for debugging and cuda utilization
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Initialize the PPO Trainer instance, add the dataset, model, tokenizer and PPOConfig object

ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model,
    tokenizer,
    dataset=subreddit_question_dataset,
    data_collator=collator
)

# Set up the accelerator, connect it to the trainer instance, cuda
device = ppo_trainer.accelerator.device

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else 'cpu'


# Loading the reward model, reward tokenizer
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name).to('cuda'), AutoTokenizer.from_pretrained(reward_name)




generation_kwargs_version2 = {
    'min_length': -1,
    'top_k': 32,
    'top_p': 0.9,
    'do_sample': True,
    'max_new_tokens': 512,
    # 'temperature':0.1,
    # 'pad_token_id': tokenizer.eos_token_id
}


generation_kwargs_for_neg_KL_version2 = {
    "min_length": -1, 
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True, 
    "pad_token_id": tokenizer.eos_token_id, 
    "eos_token_id": -1,
    "max_new_tokens": 64,
    "temperature": 0.9
}

def get_reward(batch_question, batch_response):
    outputs = []
    for i in range(len(batch_question)):
        p = batch_question[i]
        resp = batch_response[i]
        inputs = reward_tokenizer(p, resp, return_tensors='pt').to('cuda')
        score = reward_model(**inputs).logits[0].cpu().detach()
        outputs.append(score)

    return outputs


# Model training

stat_list = []
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(query.squeeze(), **generation_kwargs_for_negKL)

        response_tensors.append(response.squeeze()[-64:])
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    rewards = get_reward(batch['query'], batch['response'])

    stats = ppo_trainer.step(
        query_tensors,
        response_tensors,
        rewards
    )
    stat_list.append({'stats': stats})

    ppo_trainer.log_stats(stats, batch, rewards)


# Save the model
ppo_trainer.save_pretrained("/content/drive/MyDrive/NLP/PPO_Iteration1_Dberta_2000_may15_1AM")

# reload with generator, args
gen_for_inference = {'min_length': -1,
 'top_k': 0.0,
 'top_p': 1.0,
 'do_sample': True,
 'pad_token_id': 50256,
 'eos_token_id': -1,
 'max_new_tokens': 128,
 'temperature': 0.9}

generator_for_bm = pipeline('text-generation', model='/content/drive/MyDrive/NLP/SFT_GPT-2M_Dolly15k', **gen_for_inference)
generator_for_ppo = pipeline('text-generation', model='/content/drive/MyDrive/NLP/PPO_Iteration1_Dberta_2000_may15_1AM')

for i in range(10):
    q = subreddit_question_dataset[i]['query']
    print('PPO:',generator_for_ppo(q, **gen_for_inference))
    print('\n')
    print('BM:',generator_for_bm(q, **gen_for_inference))
    print('*************')


# Save the statistics of the model
temp = []
for s in stat_list:
    temp.append(s['stats'])


for t in temp:
    for key in t.keys():
        if isinstance(t[key], np.ndarray):
            t[key] = t[key].tolist()

with open("/content/drive/MyDrive/NLP/PPO_Iteration1_Dberta_2000_may15_1AM-STATS.json", "w") as outfile: 
    json.dump({'stats':temp}, outfile)


# plt.plot(range(len(kl_coeff)), kl_coeff, label='KL Coeff')
# plt.xlabel('Time Steps (Epochs)')
# plt.ylabel('kl_coeff')
# plt.title('kl_coeff Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(range(len(policy_loss)), policy_loss, label='policy_loss ')
# plt.xlabel('Time Steps (Epochs)')
# plt.ylabel('policy_loss')
# plt.title('pol loss Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.plot(range(len(value_loss)), value_loss, label='Value Loss')
# plt.xlabel('Time Steps (Epochs)')
# plt.ylabel('Value Loss')
# plt.title('Value Loss Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()

ppo_trainer.save_pretrained("ppo_trial_1")
generator_for_ppo = pipeline('text-generation', model='ppo_trial_1')

# use the above code for testing generations

# generator_SFT_PPO = pipeline('text-generation', model='/content/drive/MyDrive/NLP/ppo_trial_1')
# generator_BM_PPO = pipeline('text-generation', '/content/drive/MyDrive/NLP/ppo_trial_2_base_model')

# cnt = 0
# for data in subreddit_question_dataset:
#     print(data['query'])
#     print('Base+PPO' ,generator_BM_PPO(data['query'])[0]['generated_text'])
#     print('SFT+PPO', generator_SFT_PPO(data['query'])[0]['generated_text'])

#     print('________')
#     cnt +=1
#     if cnt == 15:
#         break