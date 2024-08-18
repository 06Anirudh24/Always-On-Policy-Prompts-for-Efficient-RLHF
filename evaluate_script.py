import os
import torch
import argparse
import json
os.system("pip install datasets")
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import sys

def gemini_setup(api_key):
    os.system("pip install -q -U google-generativeai")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    glm_config = genai.GenerationConfig(temperature=0.99)
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    gemini_model = genai.GenerativeModel('gemini-pro', generation_config=glm_config, safety_settings=safety_settings)
    return gemini_model

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def get_prompts(dataset_name, no_of_prompts):
    ds = load_dataset(dataset_name, split="validation")
    dup_prompts = ds['prompt']
    prompts = []
    for i in dup_prompts:
        if len(prompts) < no_of_prompts and i not in prompts:
            prompts.append(str(i))
    return prompts

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_prompts(model, tokenizer, prompts, dataset_name, model_name):
    print("generations for model {}======================================>".format(model_name))
    tokenizer.pad_token = tokenizer.eos_token
    maxlength = 700
    batch_size = 10
    num_batches = len(prompts) // batch_size + (len(prompts) % batch_size > 0)
    generated_responses = []
    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]
        encoded_prompts = tokenizer(batch_prompts, return_tensors='pt', max_length=maxlength, padding=True, truncation=True)
        model.to(device)
        encoded_prompts = {key: value.to(device) for key, value in encoded_prompts.items()}
        with torch.no_grad():
            output = model.generate(
                top_k=0.0,
                top_p=1.0,
                input_ids=encoded_prompts['input_ids'],
                attention_mask=encoded_prompts['attention_mask'],
                max_length=1024,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.99,
                pad_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            generated_responses.extend([tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(output))])

        torch.cuda.empty_cache()
    stats={}
    only_generations=[ i[len(prompt):].strip() for prompt,i in zip(prompts,generated_responses)]
    lens= [len(i) for i in only_generations]
    stats['max'] = max(lens)
    stats['min'] =min(lens)
    stats['mean'] = sum(lens)/len(lens)
    with open("{}_generation_length".format(model_name), "w") as f:
        json.dump(stats, f, indent=4)
        print("Generations length saved.")
    prompt_responses = {
        prompt: response[len(prompt):].strip()
        for prompt, response in zip(prompts, generated_responses)
    }
    output_file_name = f"{model_name}_{dataset_name}_generations.json"
    with open(output_file_name, "w") as f:
        json.dump(prompt_responses, f, indent=4)
    print("Generated responses saved to:", output_file_name)
    return output_file_name


def llm_evaluate(file1, file2, model1_name, model2_name, gemini_model):
    def load_data_from_json(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    data1 = load_data_from_json(file1)
    data2 = load_data_from_json(file2)
    prompts = [pair[0] for pair in list(data1.items())]
    responses_a = [pair[1] for pair in list(data1.items())]
    responses_b = [pair[1] for pair in list(data2.items())]
    prompt_template = ("Imagine you are an evaluator who is evaluating answers. You need to evaluate two potential responses"
                       "to determine which one is more helpful in resolving "
                       "your issue or following the guidance provided . Consider which response provides "
                       "the most practical, informative, and supportive guidance for your situation. Question/task: {}, Response A: {}, "
                       "Response B: {}. Return answer as single letter : A or B. Do not add any additional text to the answer. ")
    generated_responses = []
    for i in range(len(prompts)):
        response = gemini_model.generate_content(prompt_template.format(prompts[i], responses_a[i], responses_b[i]))
        generated_responses.append(response.text)
    output_file = f"{model1_name}_{model2_name}_llm_evaluations.json"
    with open(output_file, "w") as f:
        json.dump(generated_responses, f, indent=4)
    print("Results saved to:", output_file)
    return output_file

def perplexity(model_path, prompts, model_name):
    os.system("pip install evaluate")
    from evaluate import load
    perplexity = load("perplexity", module_type="metric")
    prompts_=[i for i in prompts if len(i)<1024]
    results = perplexity.compute(predictions=prompts, model_id=model_path)
    output_file = f"{model_name}_perplexity.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to:", output_file)
def toxicity(model_path, prompts, model_name):
  os.system("pip install evaluate")
  from evaluate import load
  toxicity = load("toxicity", model_path,module_type="measurement")
  prompts_=[i for i in prompts if len(i)<1024]
  results = toxicity.compute(predictions=prompts)
  output_file="{}_toxicity.json".format(model_name)
  with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
  print("Results saved to:", output_file)

def win_rate(file1, model1_name, model2_name):
    with open(file1, 'r') as f:
        data = json.load(f)
    counter = Counter(data)
    count_a = counter['A']
    count_b = counter['B']
    winrate_a = f"Win Rate of {model1_name} over {model2_name} = {count_a / len(data):.2f}"
    winrate_b = f"Win Rate of {model2_name} over {model1_name} = {count_b / len(data):.2f}"
    print(winrate_a)
    print(winrate_b)
    output_file = f"{model1_name}_{model2_name}_winrates.json"
    with open(output_file, "w") as f:
        json.dump([winrate_a, winrate_b], f, indent=4)
    print("Results saved to:", output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="gpt2")
    parser.add_argument("--model2", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--no_of_records", type=int, default=250)
    parser.add_argument("--metric", choices=["llm", "perplexity", "toxicity","all"], required=True)
    parser.add_argument("--api_key", type=str, required=True, help="API key for Gemini model")
    args = parser.parse_args()
    model1_name = args.model1.split("/")[-1]
    model2_name = args.model2.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    model_name = model1_name
    if args.metric == "all":
        if args.model2 is None:
            print("Cannot calculate win rate without model2")
            sys.exit()

        gemini_model = gemini_setup(args.api_key)
        model1, tokenizer1 = load_model(args.model1)
        model2, tokenizer2 = load_model(args.model2)
        prompts = get_prompts(args.dataset, args.no_of_records)
        generations_model1 = generate_prompts(model1, tokenizer1, prompts, dataset_name,model1_name)
        generations_model2 = generate_prompts(model2, tokenizer2, prompts, dataset_name,model2_name)
        output_file = llm_evaluate(generations_model1, generations_model2, model1_name, model2_name, gemini_model)
        win_rate(output_file, model1_name, model2_name)
        perplexity(args.model1, prompts, model1_name)
        perplexity(args.model2, prompts, model2_name)

    elif args.metric == "llm":
        if args.model2 is None:
            print("Cannot calculate win rate without model2")
            sys.exit()

        gemini_model = gemini_setup(args.api_key)
        model1, tokenizer1 = load_model(args.model1)
        model2, tokenizer2 = load_model(args.model2)
        model1_name = args.model1.split("/")[-1]
        model2_name = args.model2.split("/")[-1]
        dataset_name = args.dataset.split("/")[-1]
        prompts = get_prompts(args.dataset, args.no_of_records)
        generations_model1 = generate_prompts(model1, tokenizer1, prompts, dataset_name,model1_name)
        generations_model2 = generate_prompts(model2, tokenizer2, prompts, dataset_name,model2_name)
        output_file = llm_evaluate(generations_model1, generations_model2, model1_name, model2_name, gemini_model)
        win_rate(output_file, model1_name, model2_name)

    elif args.metric == "perplexity":
        model, tokenizer = load_model(args.model1)
        prompts = get_prompts(args.dataset, args.no_of_records)
        perplexity(args.model1, prompts, args.model1.split("/")[-1])

    elif args.metric == "toxicity":
        model, tokenizer = load_model(args.model1)
        prompts = get_prompts(args.dataset, args.no_of_records)
        generations = generate_prompts(model,tokenizer,prompts,dataset_name,model_name)
        generations = json.load(open(generations))
        toxicity(args.model1, generations, args.model1.split("/")[-1])

if __name__ == "__main__":
    main()
