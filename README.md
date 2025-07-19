# NLP Fine-Tuning Project

This repository contains code for fine-tuning a language model on a custom dataset using HuggingFace Transformers.

## What I Learned & Accomplished
- Fine-tuned GPT-2 on custom conversational data to build a personalized chatbot with coherent and context-aware responses.

- Developed efficient data preprocessing and tokenization workflows, carefully managing context and special tokens (e.g., EOS) to improve output quality.

- Navigated hardware and resource limitations by optimizing training setups, such as skipping train-test splits initially to speed up experimentation and model tuning.

- Leveraged HuggingFace Transformers’ generation parameters (like max_new_tokens, temperature, top_p, and repetition_penalty) to balance response creativity and prevent rambling.

- Integrated attention masks effectively to ensure reliable model input handling and reduce unexpected generation behavior.

- Explored summarization techniques to dynamically condense conversation context, helping maintain coherence within limited input size constraints.

- Gained practical knowledge of NLP fundamentals, including tokenization, decoding strategies, and handling of special tokens.

- Enhanced software engineering skills by structuring modular scripts, writing clear documentation, and managing an iterative training pipeline under real-world constraint

## Project Structure

- `data_parser.py` — Script to preprocess and format the raw input data.
- `train.py` — Script to fine-tune the language model on the processed data.
- `dialogue_train.txt` — (Optional) Example input data file to test the parsing script.

## Status

🚧 Work in progress — the model is fine-tuned and you can generate outputs by providing prompts,  
but a formal test set split and automated evaluation are planned for future updates.

## Requirements

- Python 3.8+
- torch
- transformers
- datasets (optional)
