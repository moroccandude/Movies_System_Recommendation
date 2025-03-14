from flask import Flask, jsonify, request
import requests
import json
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# URL of your Elasticsearch instance
ES_URL = "http://localhost:9200"
ES_INDEX_RECOMMENDATIONS = "movie_recommendations"
ES_INDEX_MOVIES = "movies"

# Create an index in Elasticsearch (if it doesn't exist)
@app.route('/create_index', methods=['POST'])
def create_index():
    index_name = request.json.get('index_name')
    if not index_name:
        return jsonify({"error": "Index name is required"}), 400

    # Create index using Elasticsearch REST API (via requests)
    url = f"{ES_URL}/{index_name}"
    response = requests.put(url)
    
    if response.status_code == 200:
        return jsonify({"message": f"Index '{index_name}' created successfully!"}), 201
    else:
        return jsonify({"error": response.json()}), response.status_code

# Index a document in Elasticsearch
@app.route('/index_document', methods=['POST'])
def index_document():
    index_name = request.json.get('index_name')
    document = request.json.get('document')
    document_id = request.json.get('id')

    if not index_name or not document:
        return jsonify({"error": "Index name and document are required"}), 400

    if document_id:
        url = f"{ES_URL}/{index_name}/_doc/{document_id}"
    else:
        url = f"{ES_URL}/{index_name}/_doc"
    
    response = requests.post(url, json=document)
    
    if response.status_code in [200, 201]:
        return jsonify({"message": "Document indexed successfully!", "id": response.json()['_id']}), 201
    else:
        return jsonify({"error": response.json()}), response.status_code

# Search for a document in Elasticsearch
@app.route('/search', methods=['GET'])
def search():
    index_name = request.args.get('index_name')
    query = request.args.get('query')
    field = request.args.get('field', 'movie_title')  # Default to searching movie titles

    if not index_name or not query:
        return jsonify({"error": "Index name and query are required"}), 400

    url = f"{ES_URL}/{index_name}/_search"
    search_query = {
        "query": {
            "match": {
                field: query
            }
        }
    }
    
    response = requests.post(url, json=search_query)
    
    if response.status_code == 200:
        hits = response.json()['hits']['hits']
        results = [hit['_source'] for hit in hits]
        return jsonify({"results": results, "count": len(results)}), 200
    else:
        return jsonify({"error": response.json()}), response.status_code

# Get a document by its ID from Elasticsearch
@app.route('/get_document/<index_name>/<document_id>', methods=['GET'])
def get_document(index_name, document_id):
    # URL to retrieve the document from Elasticsearch
    url = f"{ES_URL}/{index_name}/_doc/{document_id}"
    
    # Make a GET request to Elasticsearch
    response = requests.get(url)
    
    if response.status_code == 200:
        # Document found, return it
        document = response.json()['_source']
        return jsonify({"document": document}), 200
    else:
        # Document not found
        return jsonify({"error": response.json()}), response.status_code

# MOVIE RECOMMENDATION SPECIFIC ENDPOINTS

# Get recommendations for a specific user
@app.route('/recommendations/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    limit = request.args.get('limit', 10, type=int)
    
    url = f"{ES_URL}/{ES_INDEX_RECOMMENDATIONS}/_search"
    query = {
        "query": {
            "term": {
                "userId": user_id
            }
        },
        "sort": [
            {"predicted_rating": {"order": "desc"}}
        ],
        "size": limit
    }
    
    try:
        response = requests.post(url, json=query)
        if response.status_code == 200:
            hits = response.json()['hits']['hits']
            
            if not hits:
                return jsonify({"message": f"No recommendations found for user {user_id}"}), 404
                
            recommendations = [hit['_source'] for hit in hits]
            return jsonify({
                "userId": user_id,
                "recommendations": recommendations,
                "count": len(recommendations)
            }), 200
        else:
            logger.error(f"Elasticsearch error: {response.text}")
            return jsonify({"error": "Failed to retrieve recommendations"}), response.status_code
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Get movie details
@app.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    url = f"{ES_URL}/{ES_INDEX_MOVIES}/_search"
    query = {
        "query": {
            "term": {
                "movie_id": movie_id
            }
        }
    }
    
    try:
        response = requests.post(url, json=query)
        if response.status_code == 200:
            hits = response.json()['hits']['hits']
            
            if not hits:
                return jsonify({"message": f"Movie {movie_id} not found"}), 404
                
            movie = hits[0]['_source']
            return jsonify({"movie": movie}), 200
        else:
            logger.error(f"Elasticsearch error: {response.text}")
            return jsonify({"error": "Failed to retrieve movie details"}), response.status_code
    except Exception as e:
        logger.error(f"Error retrieving movie: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Search movies by title, genre, etc.
@app.route('/movies/search', methods=['GET'])
def search_movies():
    query = request.args.get('query')
    field = request.args.get('field', 'movie_title')
    limit = request.args.get('limit', 10, type=int)
    
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
        
    url = f"{ES_URL}/{ES_INDEX_MOVIES}/_search"
    search_query = {
        "query": {
            "match": {
                field: query
            }
        },
        "size": limit
    }
    
    try:
        response = requests.post(url, json=search_query)
        if response.status_code == 200:
            hits = response.json()['hits']['hits']
            results = [hit['_source'] for hit in hits]
            return jsonify({
                "results": results, 
                "count": len(results),
                "query": query,
                "field": field
            }), 200
        else:
            logger.error(f"Elasticsearch error: {response.text}")
            return jsonify({"error": "Failed to search movies"}), response.status_code
    except Exception as e:
        logger.error(f"Error searching movies: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Get top-rated movies (based on predicted ratings)
@app.route('/recommendations/top_movies', methods=['GET'])
def get_top_movies():
    limit = request.args.get('limit', 20, type=int)
    
    url = f"{ES_URL}/{ES_INDEX_RECOMMENDATIONS}/_search"
    query = {
        "query": {
            "match_all": {}
        },
        "sort": [
            {"predicted_rating": {"order": "desc"}}
        ],
        "size": limit
    }
    
    try:
        response = requests.post(url, json=query)
        if response.status_code == 200:
            hits = response.json()['hits']['hits']
            results = [hit['_source'] for hit in hits]
            return jsonify({
                "top_movies": results, 
                "count": len(results)
            }), 200
        else:
            logger.error(f"Elasticsearch error: {response.text}")
            return jsonify({"error": "Failed to retrieve top movies"}), response.status_code
    except Exception as e:
        logger.error(f"Error retrieving top movies: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Get similar users (users who might have similar taste)
@app.route('/recommendations/similar_users/<int:user_id>', methods=['GET'])
def get_similar_users(user_id):
    limit = request.args.get('limit', 5, type=int)
    
    # First get this user's top rated movies
    url = f"{ES_URL}/{ES_INDEX_RECOMMENDATIONS}/_search"
    query = {
        "query": {
            "term": {
                "userId": user_id
            }
        },
        "sort": [
            {"predicted_rating": {"order": "desc"}}
        ],
        "size": 10  # Get top 10 movies for this user
    }
    
    try:
        response = requests.post(url, json=query)
        if response.status_code != 200 or not response.json()['hits']['hits']:
            return jsonify({"message": f"No data found for user {user_id}"}), 404
            
        # Get top movies for this user
        top_movies = [hit['_source']['movieId'] for hit in response.json()['hits']['hits']]
        
        # Now find other users who also like these movies
        similar_users_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"movieId": top_movies}},
                        {"range": {"predicted_rating": {"gte": 4.0}}}  # Only high ratings
                    ],
                    "must_not": [
                        {"term": {"userId": user_id}}  # Exclude the original user
                    ]
                }
            },
            "aggs": {
                "similar_users": {
                    "terms": {
                        "field": "userId",
                        "size": limit
                    }
                }
            },
            "size": 0  # We only care about the aggregation
        }
        
        similar_response = requests.post(url, json=similar_users_query)
        if similar_response.status_code != 200:
            return jsonify({"error": "Failed to find similar users"}), similar_response.status_code
            
        similar_users = [
            {"userId": bucket["key"], "common_movies": bucket["doc_count"]} 
            for bucket in similar_response.json()['aggregations']['similar_users']['buckets']
        ]
        
        return jsonify({
            "userId": user_id,
            "similar_users": similar_users,
            "count": len(similar_users)
        }), 200
        
    except Exception as e:
        logger.error(f"Error finding similar users: {str(e)}")
        return jsonify({"error": "cannot find similar users"})
    
if __name__=="__main__":
 app.run(debug=True)