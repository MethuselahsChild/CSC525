import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_scheduler
from torch.utils.data import DataLoader, TensorDataset

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# Function to load data from a file
def load_data(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# Function to create a dataset
def create_dataset(text, tokenizer):
    encodings = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'])

# Training function
def train(model, dataset, device, epochs=80):
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Function to save the model and tokenizer
def save_model(model, tokenizer, directory="trained_model"):
    model_path = os.path.join(os.path.dirname(__file__), directory)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    print(f"Model and tokenizer saved to {model_path}")

# Main execution block
if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.to(device)

    # Load the text data
    text_data = load_data('eightdates.txt')

    # Create dataset
    dataset = create_dataset(text_data, tokenizer)

    # Train the model
    train(model, dataset, device)

    # Save the model and tokenizer
    save_model(model, tokenizer)
