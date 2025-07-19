# NLP Fine-Tuning Project

This repository contains code for fine-tuning a language model on a custom dataset using HuggingFace Transformers.

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
