import os
from openai import OpenAI
import json
import mapbox
import requests
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


mapbox_access_token = os.getenv("pk.eyJ1IjoibGF6YXJ1czExIiwiYSI6ImNsdW9zNWlzZTFmdDgyamxuNHh3dzgwdHgifQ.AfBlQPuXlwdZXUxxk4yXuA")
geocoder = mapbox.Geocoder(access_token=mapbox_access_token)


api_key = "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"
client = OpenAI(api_key=api_key)

data_storage_path = "chatbot_feedback.json"


def fetch_local_data(zip_code, place_type='restaurant'):
    """
    Fetches data from Mapbox for local places based on the specified zip code and place type.
    
    Args:
        zip_code (str): The user's zip code.
        place_type (str): Type of place to search for (e.g., 'restaurant', 'park', 'museum').
    
    Returns:
        list: A list of dictionaries, each representing a place.
    """

    response = geocoder.forward(zip_code)
    
    if response.status_code != 200 or not response.geojson()['features']:
        print("Could not find location for the provided zip code.")
        return []

    feature = response.geojson()['features'][0]
    longitude, latitude = feature['geometry']['coordinates']

    search_url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{place_type}.json"
    params = {
        'access_token': mapbox_access_token,
        'proximity': f"{longitude},{latitude}",
        'limit': 10 #number of results
    }

    response = requests.get(search_url, params=params)

    if response.status_code != 200:
        print("Failed to find places.")
        return []

    places = []
    for place in response.json()['features']:
        places.append({
            'name': place['text'],
            'address': place['place_name'],
            'rating': 'N/A',
            'place_id': place['id']
        })
    return places

def chat_with_gpt(prompt, temperature=0.7, max_tokens=150):
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    response = chat_completion.choices[0].message.content 
    return response

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model():
    if not os.path.exists(data_storage_path):
        print("No data available for training.")
        return None

    with open(data_storage_path, 'r') as file:
        data_log = json.load(file)

    features = np.array([len(item['user_input']) for item in data_log]).reshape(-1, 1)
    labels = np.array([item['user_feedback'] for item in data_log])

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(features_train, labels_train, epochs=10, validation_data=(features_test, labels_test))

    return model

def ml_learning(user_input, chatbot_response, user_feedback):
    if os.path.exists(data_storage_path):
        with open(data_storage_path, 'r') as file:
            data_log = json.load(file)
    else:
        data_log = []

    data_log.append({
        'user_input': user_input,
        'chatbot_response': chatbot_response,
        'user_feedback': int(user_feedback)
    })

    with open(data_storage_path, 'w') as file:
        json.dump(data_log, file, indent=4)

def user_interaction():
    print("Welcome to the Date Chatbot! Type 'quit' to exit.")
    print("Feel free to ask me for advice or questions.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Thank you for chatting. Goodbye!")
            break

        chatbot_response = chat_with_gpt(user_input)
        print("Chatbot:", chatbot_response)

if __name__ == "__main__":
    user_interaction()
