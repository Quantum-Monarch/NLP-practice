from operator import truediv

from transformers import GPT2Tokenizer, GPT2LMHeadModel,pipeline
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from parserm import data2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad_token by default


model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
toy_data = data2



separator_token = "<|sep|>"
tokenizer.add_special_tokens({'additional_special_tokens': [separator_token]})
model.resize_token_embeddings(len(tokenizer))
class ToyDialogueDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128):
        self.examples = []

        for context, prompt, response in pairs:
            text = f" {context} {separator_token} {prompt} {tokenizer.eos_token} {response} {tokenizer.eos_token}"
            encodings_input = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            text2=f"{context} {separator_token} {prompt} {tokenizer.eos_token}"
            enc_prompt = tokenizer(
                text2,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            input__ids=encodings_input["input_ids"].squeeze()
            labelz = input__ids.clone()
            labelz[: enc_prompt["input_ids"].ne(tokenizer.pad_token_id).sum()] = -100
            labelz[input__ids == tokenizer.pad_token_id] = -100

            self.examples.append({
                "input_ids": encodings_input["input_ids"].squeeze(),
                "attention_mask": encodings_input["attention_mask"].squeeze(),
                "labels": labelz
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: v for k, v in self.examples[idx].items()}

dataset = ToyDialogueDataset(toy_data, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True)



optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
epochs = 5


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Loss: {loss.item():.4f}")

model.eval()
context=""
prompt = "Hi, how are you?"
text = f" {context} {separator_token} {prompt}{tokenizer.eos_token} "
input_ids= tokenizer.encode(text,return_tensors="pt").to(device)

sample_output = model.generate(
    input_ids,
    max_new_tokens=50,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    )

print("Response:", tokenizer.decode(sample_output[0], skip_special_tokens=True))

