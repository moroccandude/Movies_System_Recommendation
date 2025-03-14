import requests
import os
from dotenv import load_dotenv
from confluent_kafka import Producer
import json
import requests
import time

# Kafka configuration
KAFKA_BROKER = "localhost:9092"  # Ensure this matches your Kafka container settings
TOPIC_NAME = "users"

# Configure the Kafka Producer
producer_config = {
    "bootstrap.servers": KAFKA_BROKER,
    "acks": "all",  # Ensure all replicas acknowledge the message
      "retries": 3  # Retries in case of transient failures
}

producer = Producer(producer_config)

def get_all_movies(api_key, page=1, max_pages=5):
    """
    Fetch movies from TMDB API
    
    Args:
        api_key (str): Your TMDB API key
        page (int): Starting page number
        max_pages (int): Maximum number of pages to fetch
    
    Returns:
        list: List of movie dictionaries
    """
    base_url = "https://api.themoviedb.org/3"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json;charset=utf-8"
    }
    
    all_movies = []
    current_page = page
    
    while current_page <= max_pages:
        # Get popular movies (you can change this endpoint as needed)
        endpoint = f"{base_url}/movie/popular"
        params = {
            "language": "en-US",
            "page": current_page
        }
        
        response = requests.get(endpoint, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            movies = data.get("results")
            # all_movies.extend(movies)
            
            # Check if we've reached the last page
            if current_page >= data.get("total_pages", 0):
                break
                
            current_page += 1
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
    
    return movies

def main():
    load_dotenv()

    # Your API key
    api_key=os.getenv("API_KEY")

    # Get movies
    movies = get_all_movies(api_key)
    print(movies)
    # Print movie titles
    print(f"Found {len(movies)} movies:")
    for i, movie in enumerate(movies, 1):
       q={ "movie":movie['title'],"date":movie['release_date'],"Rating": movie['vote_average'],'vote_average':movie['vote_average'], 'vote_count': movie['vote_count'],'popularity':movie['popularity']}
       producer.produce(TOPIC_NAME, key=str(user["login"]["uuid"]+user["email"]), value=user_json, callback=delivery_report)
       time.sleep(0.6)  # Small delay to simulate a real-time stream
       producer.flush()  # Ensure all messages are sent before exiting


if __name__ == "__main__":
    main()