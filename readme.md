## Fine-Tuning Intelligence: Exploring Dynamic Prompts in RLHF for Improved Alignment

**Introduction**

This class project explores a novel approach to Reinforcement Learning from Human Feedback (RLHF) for LLMs, with the objective of improving LLM alignment with human values by utilizing a dynamic fine-tuning prompt set based on the model's performance during training, i.e. an on-policy prompt dataset. 

**Traditional RLHF Limitations**

Standard RLHF typically uses static prompts during fine-tuning. This approach can be inefficient as the model wastes iterations on prompts it's already good at, neglecting areas where alignment needs improvement.

**Proposed Approach: Dynamic, On-Policy Fine-Tuning**

This project proposes a modification to to the standard RLHF pipeline that leverages dynamic fine-tuning prompts. This dataset is generated based on the model's intermediate performance, focusing training efforts on areas where alignment is lacking, using the help of some of the current SOTA LLMs.

**Experiments and Results**

Three models were trained with the same number of steps and data records:

1. **Vanilla RLHF:** Trained on synthetically generated data using a static prompt set.
2. **Starts-On-Policy (SOP):** Trained on a synthetic dataset based on the model's initial performance.
3. **Always-On-Policy (AOP):** Trained with a dynamic performance-based dataset that changes with each RLHF iteration.

The AOP model achieved significant improvements compared to the base model, the SFT model, and even the vanilla RLHF and SOP models.

**Significance**

This project demonstrates the effectiveness of dynamic fine-tuning prompts for RLHF in LLMs. Even with limitations in compute and time, the results are promising and offer a novel approach to efficiently utilizing human feedback for LLM alignment.

**Project Files**

1. **Reward_Models_trial.ipynb:** Explores and compares different reward models.
2. **Method_1_Dataset_Generation.ipynb:** Generates a synthetic dataset for the SOP model using SHP and Gemini.
3. **Dynamic_Data_Generation.ipynb:** Computes responses, reward scores, and generates new questions for the AOP model.
4. **dataset.ipynb:** Defines a dataset class to prepare the synthetic data for RLHF training.
5. **SFT.ipynb:** Trains a GPT-2 model using Dolly 15K data.
6. **ppo_training_version1.py:** Script for vanilla RLHF model training.
7. **ppo_training_version2.py:** Script for SOP model training.
8. **RLHF_NLP_Project_Method2.ipynb:** Script for AOP model training.
9. **evaluate_script.py:** Script for evaluating model performance (helpfulness, toxicity, perplexity, winrate).
10. **training_graphs.ipynb:** Script for plotting training results.