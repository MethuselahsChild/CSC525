#Ensure that you have all the required libraries installed (NLTK, PyTorch, scikit-learn, requests, googlemaps, json, datetime).
#Enter Google Maps API key on line 136 = (from .doc or .txt file, replace 'XXXXXXXXXXXXXXXXXXXXXXXX)
#Place the date ideas dataset file (dateideas.txt) in the same directory as the script.

import os
import json
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import googlemaps
from datetime import datetime, timedelta
from googlemaps.exceptions import ApiError, HTTPError, TransportError
import re
import random

# Set up NLTK = remove # from the following lines if needed downloaded on your machine.
#nltk.download('punkt', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('wordnet', quiet=True)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main_menu():
    print("1. Train model")
    print("2. Start chatbot")
    choice = input("Enter your choice: ")
    return choice

tokenizer = word_tokenize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class DateIdeasDataset(torch.utils.data.Dataset):
    def __init__(self, filename, max_seq_length):
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences = text.split('.')
        self.data = [self.preprocess_sentence(sentence) for sentence in sentences]
        self.labels = self.compute_labels(sentences)
        self.word_to_index = {}
        for sentence in self.data:
            for word in sentence.split():
                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)
        self.max_seq_length = max_seq_length

    def preprocess_sentence(self, sentence):
        words = tokenizer(sentence.lower())
        words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    def compute_labels(self, sentences):
        return [0] * len(sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx]
        label = self.labels[idx]
        input_tensor = torch.tensor([self.word_to_index.get(word, 0) for word in input_text.split()])
        padded_tensor = torch.zeros(self.max_seq_length, dtype=torch.long)
        padded_tensor[:len(input_tensor)] = input_tensor[:self.max_seq_length]
        return padded_tensor, label

class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, num_classes=5, max_seq_length=100):
        super(SimpleClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * max_seq_length, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=100):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {100 * correct / total}%")
    return model

def save_model(model, vocab_size, filename='chatbot_model.pth'):
    checkpoint = {'state_dict': model.state_dict(), 'vocab_size': vocab_size}
    torch.save(checkpoint, filename)

def load_model(model_class, filename='chatbot_model.pth'):
    try:
        checkpoint = torch.load(filename)
        model = model_class(vocab_size=checkpoint['vocab_size'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    except FileNotFoundError:
        print(f"Model file '{filename}' not found.")
        return None

gmaps = googlemaps.Client(key='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

def fetch_place_details(place_id):
    try:
        fields = ['name', 'formatted_address', 'formatted_phone_number', 'rating', 'review', 'url']
        details = gmaps.place(place_id=place_id, fields=fields)
        return details.get('result', {})
    except Exception as e:
        return {"error": str(e)}

def format_event_list(events, location, num_events=10):
    if not events:
        return f"Sorry, I couldn't find any activities in {location}. Please try a different city."
    selected_events = random.sample(events, min(num_events, len(events)))
    event_list_text = "\n".join([f"{index+1}. {event['name']} - {event['type']} (Rated: {event['rating']})" for index, event in enumerate(selected_events)])
    return f"Here are some Date Night ideas I found in {location}:\n{event_list_text}\nYou can ask me for more information about these activities by typing their name or number."

def detailed_event_info(event):
    details = fetch_place_details(event['place_id'])
    info = (f"About {event['name']}:\n"
            f"- Address: {details.get('formatted_address', 'Address not available')}\n"
            f"- Phone: {details.get('formatted_phone_number', 'Phone number not available')}\n"
            f"- Rating: {event['rating']}\n"
            f"- Reviews: {' '.join(review['text'] for review in details.get('reviews', []))}\n"
            f"- More Info: {details.get('url', 'No URL available')}")
    return info

def fetch_events(location, page_token=None):
    try:
        geocode_result = gmaps.geocode(location)
        if not geocode_result:
            return [], "Failed to find the location", None
        latlng = geocode_result[0]['geometry']['location']
        lat, lng = latlng['lat'], latlng['lng']

        # Search queries
        queries = [
            "restaurant, cafe, bar, pub, night_club",
            "bowling_alley, movie_theater, amusement_park, aquarium, art_gallery, museum, zoo",
            "spa, scenic_view, park, shopping_mall, theater, concert, event, festival",
            "winery, comedy_club, dance_club, escape_room, arcade, karaoke",
            "mini_golf, paintball, go_kart, ice_skating, rock_climbing, trampoline_park, laser_tag",
            "cooking_class, pottery_class, art_class, wine_tasting, beer_tasting, food_tour",
            "boat_tour, helicopter_tour, hot_air_balloon, horse_riding, picnic_spot"
        ]
        events = []
        place_ids = set()
        for query in queries:
            while len(events) < 10:
                response = gmaps.places(query=query, location=(lat, lng), radius=50, page_token=page_token)
                if not response.get('results'):
                    break
                for result in response['results']:
                    place_id = result.get('place_id', '')
                    if place_id not in place_ids:  # Check if the place ID is unique
                        url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                        types_list = result.get('types', [])
                        formatted_types = ", ".join([t for t in types_list if 'point_of_interest' not in t])
                        event = {
                            'name': result['name'],
                            'type': formatted_types.capitalize(),
                            'rating': result.get('rating', 'Rating not available'),
                            'url': url,
                            'place_id': place_id
                        }
                        events.append(event)
                        place_ids.add(place_id)
                    if len(events) >= 10:
                        break
                page_token = response.get('next_page_token', None)
                if not page_token:
                    break
        return events, None, page_token
    except Exception as e:
        return [], str(e), None

def find_activity_by_name(input_text, events):
    if input_text.isdigit():
        line_number = int(input_text)
        if 1 <= line_number <= len(events):
            return events[line_number - 1]
    for event in events:
        if input_text.lower() in event['name'].lower():
            return event
    return None

def chat_with_user(model, dataset):
    print("Welcome to the Date Night Chatbot! Please tell me your city to find activities.")
    user_location = ""
    next_page_token = None
    context = {}
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
            print("Chatbot: Thank you for using the Date Night Chatbot. Have a great day!")
            break
        response = generate_response(model, dataset, user_input, context)
        print(f"Chatbot: {response}")

def generate_response(model, dataset, input_text, context):
    words = word_tokenize(input_text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    if 'switch' in input_text.lower() or 'change' in input_text.lower():
        context.pop('location', None)
        return "Sure! Which city would you like to explore?"

    if not context.get('location'):
        context['location'] = input_text.strip()
        events, error, next_page_token = fetch_events(context['location'])
        if error:
            context.pop('location', None)
            return f"I'm sorry, I couldn't find any events in {input_text}. Please try a different city."
        context['events'] = events
        context['next_page_token'] = next_page_token
        context['more_info_given'] = False
        return format_event_list(events, context['location']) + "\nWould you like information about another activity? Or you can ask for more activities."

    # User asks for more activities
    if 'more' in input_text.lower():
        events, error, next_page_token = fetch_events(context['location'], context.get('next_page_token'))
        if error:
            return "I'm sorry, I couldn't find any more events. Please try another city."
        context['events'] = events
        context['next_page_token'] = next_page_token
        context['more_info_given'] = False
        return format_event_list(events, context['location']) + "\nWould you like information about another activity? Or you can ask for more activities."

    # Handle specific activity queries
    selected_event = find_activity_by_name(input_text, context['events'])
    if selected_event:
        context['more_info_given'] = True
        return detailed_event_info(selected_event) + "\nWould you like information about another activity? Or you can ask for more activities."
    if not context.get('more_info_given'):
        return format_event_list(context['events'], context['location']) + "\nWould you like more information? Or you can ask for more activities."
    return "Would you like me to find more activities for you? Just type 'more' to get another list of activities."

if __name__ == "__main__":
    max_seq_length = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DateIdeasDataset('dateideas.txt', max_seq_length=max_seq_length)
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    model = SimpleClassifier(vocab_size=len(dataset.word_to_index), embedding_dim=64, hidden_size=64, num_classes=5, max_seq_length=max_seq_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    choice = main_menu()
    if choice == '1':
        model = train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=200)
        save_model(model, vocab_size=len(dataset.word_to_index))
    elif choice == '2':
        model = load_model(SimpleClassifier)
        if model:
            chat_with_user(model, dataset)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        