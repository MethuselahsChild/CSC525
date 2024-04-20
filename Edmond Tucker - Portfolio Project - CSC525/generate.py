import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

print(torch.cuda.get_device_name(0))

class Chatbot:
    def __init__(self, model_folder='trained_model', max_context_length=512):
        self.model_path = os.path.join(os.path.dirname(__file__), model_folder)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path, padding_side='left')
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_context_length = max_context_length
        self.context_tokens = []

    def reset_context(self):
        self.context_tokens = []

    def update_context(self, new_tokens):
        self.context_tokens.extend(new_tokens)
        # Trim context to only keep the most recent tokens up to max_context_length
        self.context_tokens = self.context_tokens[-self.max_context_length:]

    def ask_model(self, prompt, max_new_tokens=50):
        self.model.eval()
        input_tokens = self.tokenizer.encode(prompt, return_tensors='pt')
        self.update_context(input_tokens[0].tolist())  # Update context with the new user input

        input_ids = torch.tensor([self.context_tokens], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=30,
            top_p=0.7,
            temperature=0.5
        )

        # Update context with the generated response tokens
        generated_tokens = output_ids[0][input_ids.shape[1]:].tolist()
        self.update_context(generated_tokens)
        
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    chatbot = Chatbot()
    print("Ask your AI a question. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot.ask_model(user_input)
        print()
        print(response)
