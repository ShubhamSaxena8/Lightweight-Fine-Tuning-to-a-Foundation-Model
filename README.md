# Lightweight PEFT Fine-Tuning for Sequence Classification

This project demonstrates how to apply parameter-efficient fine-tuning (PEFT) to a pre-trained language model for a text classification task, and how to compare the performance of the base and fine-tuned models.

## Overview

The goal of this project is to:
- Load a pre-trained foundation model for sequence classification.
- Establish a baseline by evaluating the original model on a text classification dataset.
- Apply a parameter-efficient fine-tuning (PEFT) technique (LoRA) to adapt the model.
- Evaluate the fine-tuned model and compare its performance to the baseline.

The project focuses on keeping the compute and memory footprint small while still improving task-specific performance.

## Approach

1. **Model & Dataset Selection**
   - Select a pre-trained sequence classification model from Hugging Face.
   - Load a compatible text classification dataset from the `datasets` library.

2. **Baseline Evaluation**
   - Load the tokenizer and model.
   - Preprocess the dataset and run an initial evaluation to record baseline metrics (e.g., accuracy/F1).

3. **Parameter-Efficient Fine-Tuning (PEFT)**
   - Configure a LoRA-based PEFT adapter with suitable hyperparameters for the chosen model.
   - Wrap the foundation model with the PEFT configuration to create a trainable PEFT model.
   - Fine-tune the PEFT model on the training split for at least one epoch.

4. **Post-Fine-Tuning Evaluation**
   - Save the fine-tuned PEFT model.
   - Reload the PEFT model and re-run evaluation on the validation/test split.
   - Compare the metrics to the original foundation model to quantify the effect of PEFT.

## Tech Stack

- Python
- Hugging Face Transformers
- PEFT (LoRA)
- Hugging Face Datasets
- PyTorch
