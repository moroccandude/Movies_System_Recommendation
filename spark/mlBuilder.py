from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import time
import os
import requests
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename="logs.txt",filemode="w")
logger = logging.getLogger(__name__)

# Define schemas for MovieLens data
schema_record = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("item_id", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", IntegerType(), True),
])

schema_item = StructType([
    StructField("movie_id", IntegerType(), True),
    StructField("movie_title", StringType(), True),
    StructField("release_date", StringType(), True),
    StructField("video_release_date", StringType(), True),
    StructField("IMDb_URL", StringType(), True),
    StructField("unknown", IntegerType(), True),
    StructField("Action", IntegerType(), True),
    StructField("Adventure", IntegerType(), True),
    StructField("Animation", IntegerType(), True),
    StructField("Children's", IntegerType(), True),
    StructField("Comedy", IntegerType(), True),
    StructField("Crime", IntegerType(), True),
    StructField("Documentary", IntegerType(), True),
    StructField("Drama", IntegerType(), True),
    StructField("Fantasy", IntegerType(), True),
    StructField("Film-Noir", IntegerType(), True),
    StructField("Horror", IntegerType(), True),
    StructField("Musical", IntegerType(), True),
    StructField("Mystery", IntegerType(), True),
    StructField("Romance", IntegerType(), True),
    StructField("Sci-Fi", IntegerType(), True),
    StructField("Thriller", IntegerType(), True),
    StructField("War", IntegerType(), True),
    StructField("Western", IntegerType(), True),
])

# Elasticsearch API endpoint configuration
load_dotenv()
FLASK_API_URL=os.getenv("FLASK_API_URL")

def instanceSp():
    """Initialise et retourne une session Spark"""
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler=logging.StreamHandler()
    logger.addHandler(handler)
    
    try:
        logging.info("creating spark instance")
        instance_Spark = SparkSession.builder \
        .master("local[*]") \
        .appName('MovieLens ALS Recommender') \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

        return instance_Spark
    except Exception as exep :
        logging.INFO(f"failed to creating spark instance")
    

# Function to read CSV file
def reader(path: str, schema: StructType, sep: str, instance, header=False) -> DataFrame:
    """Lit un fichier CSV avec le schéma et le séparateur spécifiés"""
    logger.info(f"Reading data from {path}...")
    try:
        df = instance.read \
            .option("delimiter", sep) \
            .option("header", header) \
            .schema(schema) \
            .csv(path)
        logger.info(f"Successfully read {df.count()} records from {path}")
        return df
    except Exception as e:
        logger.error(f"Error reading file {path}: {str(e)}")
        raise


def validation_croisee_5_plis(spark, data_dir, params):
    """
    Effectue une validation croisée à 5 plis en utilisant les ensembles de données préparés
    u1.base/test à u5.base/test
    """
    resultats = []
    
    # Paramètres à tester
    rang = params.get('rang', 10)
    regParam = params.get('regParam', 0.1)
    iterations = params.get('iterations', 10)
    
    # Boucle sur les 5 plis
    for i in range(1, 6):
        logger.info(f"==== Fold {i} ====")
        train_path = os.path.join(data_dir, f"u{i}.base")
        test_path = os.path.join(data_dir, f"u{i}.test")
      
        # Lire les données d'entraînement et de test
        train_df = reader(train_path, schema_record, "\t", spark)
        test_df = reader(test_path, schema_record, "\t", spark)
         
       
        # Renommer les colonnes pour ALS
        train_df = train_df.withColumnRenamed("user_id", "userId") \
                      .withColumnRenamed("item_id", "movieId") \
                      .withColumn("rating", col("rating").cast("float"))
        test_df = test_df.withColumnRenamed("user_id", "userId") \
                     .withColumnRenamed("item_id", "movieId") \
                     .withColumn("rating", col("rating").cast("float"))
       
        # Entraîner le modèle
        model = entrainer_modele_als(train_df, rang, regParam, iterations)
        
        # Évaluer le modèle
        rmse, _ = evaluer_modele(model, test_df)
        logger.info(f"Fold {i} - RMSE: {rmse}")
        
        resultats.append(rmse)
    
    # Calculer et retourner la moyenne des RMSE
    avg_rmse = sum(resultats) / len(resultats)
    logger.info(f"Average RMSE across 5 folds: {avg_rmse}")
    
    return avg_rmse, resultats

def entrainer_modele_als(donnees_entrainement, rang=10, regParam=0.1, iterations=10):
    """
    Crée et entraîne un modèle ALS avec les paramètres spécifiés
    """
    logger.info(f"Training ALS model with rank={rang}, regParam={regParam}, iterations={iterations}")
    
    als = ALS(
        maxIter=iterations,
        regParam=regParam,
        rank=rang,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",  # Stratégie pour les nouveaux utilisateurs/films
        nonnegative=True  # Contraindre les facteurs à être non négatifs
    )
    
    model = als.fit(donnees_entrainement)
    logger.info("Model training completed")
    return model

def evaluer_modele(modele, donnees_test):
    """
    Évalue le modèle et retourne la RMSE (Root Mean Square Error)
    """
    # Génère des prédictions
    predictions = modele.transform(donnees_test)
    
    # Supprime les prédictions NaN (peut se produire avec coldStartStrategy="drop")
    predictions = predictions.filter(col("prediction").isNotNull())
    
    # Calcule la RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse", 
        labelCol="rating", 
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse, predictions

def generer_recommandations_utilisateur(modele, user_id, n_recommandations=40):
    """
    Génère des recommandations pour un utilisateur spécifique
    """
    logger.info(f"Generating {n_recommandations} recommendations for user {user_id}")
    instancea = instanceSp()
    # Créer un DataFrame avec un seul utilisateur
    users = instancea.createDataFrame([(user_id,)], ["userId"])
    
    # Générer les recommandations
    recommandations = modele.recommendForUserSubset(users, n_recommandations)
    
    # Extraire et formater les recommandations
    if recommandations.count() > 0:
        user_recs = recommandations.select(
            "userId", 
            explode("recommendations").alias("recommendation")
        )
        
        # Extraire movieId et rating de la colonne recommendations
        user_recs = user_recs.select(
            "userId",
            col("recommendation.movieId").alias("movieId"),
            col("recommendation.rating").alias("predicted_rating")
        )
        
        return user_recs
    else:
        logger.warning(f"No recommendations generated for user {user_id}")
        return None

def enrichir_recommandations(recommandations, films):
    """
    Combine les recommandations avec les informations sur les films
    """
    logger.info("Enriching recommendations with movie information")
    recommandations_enrichies = recommandations.join(
        films, 
        recommandations.movieId == films.movie_id
    ).drop(films.movie_id)
    
    return recommandations_enrichies

def create_elasticsearch_index(index_name:str):
    """Create an index in Elasticsearch using the Flask API"""
    logger.info(f"Creating Elasticsearch index: {index_name}")
    
    url = f"{FLASK_API_URL}/create_index"
    data = {"index_name": index_name}
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            logger.info(f"Successfully created index '{index_name}'")
            return True
        else:
            logger.warning(f"Failed to create index '{index_name}': {response.json()}")
            # If the index already exists, this isn't necessarily an error
            return response.status_code == 400
    except Exception as e:
        logger.error(f"Error creating Elasticsearch index: {str(e)}")
        return False

def save_recommendations_to_elasticsearch(recommendations_df, index_name, movies_df=None,):
    """
    Save recommendations to Elasticsearch using the Flask API
    
    Args:
        recommendations_df: DataFrame with recommendations
        index_name: Name of the Elasticsearch index
        movies_df: Optional DataFrame with movie information
    """
    logger.info(f"Saving recommendations to Elasticsearch index: {index_name}")
    
    # i comment it out:  because it doesn't verify Creation of index if it doesn't exist
    # create_elasticsearch_index(index_name)
    
    # Convert DataFrame to list of rows
    if movies_df and not movies_df.isEmpty():
        # Enrich recommendations with movie information if available
        enriched_recs = enrichir_recommandations(recommendations_df, movies_df)
        # Convert to list of rows
        recs_list = enriched_recs.collect()
        
        # Save each recommendation as a document
        for rec in recs_list:
            # Create document with movie information
            document = {
                "userId": rec["userId"],
                "movieId": rec["movieId"],
                "predicted_rating": float(rec["predicted_rating"]),
                "movie_title": rec["movie_title"],
                "release_date": rec["release_date"] if rec["release_date"] else "",
                "genres": {
                    "Action": rec["Action"] == 1,
                    "Adventure": rec["Adventure"] == 1,
                    "Animation": rec["Animation"] == 1,
                    "Children's": rec["Children's"] == 1,
                    "Comedy": rec["Comedy"] == 1,
                    "Crime": rec["Crime"] == 1,
                    "Documentary": rec["Documentary"] == 1,
                    "Drama": rec["Drama"] == 1,
                    "Fantasy": rec["Fantasy"] == 1,
                    "Film-Noir": rec["Film-Noir"] == 1,
                    "Horror": rec["Horror"] == 1,
                    "Musical": rec["Musical"] == 1,
                    "Mystery": rec["Mystery"] == 1,
                    "Romance": rec["Romance"] == 1,
                    "Sci-Fi": rec["Sci-Fi"] == 1,
                    "Thriller": rec["Thriller"] == 1,
                    "War": rec["War"] == 1,
                    "Western": rec["Western"] == 1
                },
                "document_type": "movie_recommendation"
            }
            
            # Index the document
            index_document(index_name, document)
    else:
        # Save recommendations without movie information
        recs_list = recommendations_df.collect()
        
        # Save each recommendation as a document
        for rec in recs_list:
            document = {
                "userId": rec["userId"],
                "movieId": rec["movieId"],
                "predicted_rating": float(rec["predicted_rating"]),
                "document_type": "movie_recommendation"
            }
            
            # Index the document
            index_document(index_name, document)

def index_document(index_name, document):
    """Index a document in Elasticsearch using the Flask API"""
    url = f"{FLASK_API_URL}/index_document"
    data = {
        "index_name": index_name,
        "document": document
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 201:
            logger.debug(f"Document indexed successfully: {document.get('userId')}-{document.get('movieId')}")
            return True
        else:
            logger.warning(f"Failed to index document: {response.json()}")
            return False
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        return False


def optimiser_hyperparametres(spark, data_dir):
    """
    Recherche les meilleurs hyperparamètres pour le modèle ALS
    """
    resultats = []
    
    # Grille de recherche pour les hyperparamètres
    rangs = [5, 10, 15, 20]
    regParams = [0.01, 0.1, 0.5, 1.0]
    iterations = 10
    
    meilleure_rmse = float('inf')
    meilleurs_params = None
    
    # Tester chaque combinaison de paramètres
    for rang in rangs:
        for regParam in regParams:
            params = {'rang': rang, 'regParam': regParam, 'iterations': iterations}
            logger.info(f"Testing parameters: {params}")
            
            # Effectuer la validation croisée
            avg_rmse, _ = validation_croisee_5_plis(spark, data_dir, params)
            
            # Enregistrer les résultats
            resultats.append((rang, regParam, avg_rmse))
            
            # Mettre à jour les meilleurs paramètres si nécessaire
            if avg_rmse < meilleure_rmse:
                meilleure_rmse = avg_rmse
                meilleurs_params = params
    
    logger.info(f"Best parameters: {meilleurs_params}, RMSE: {meilleure_rmse}")
    
    # Trier et afficher tous les résultats
    resultats_tries = sorted(resultats, key=lambda x: x[2])
    logger.info("All results (sorted by RMSE):")
    for rang, regParam, rmse in resultats_tries:
        logger.info(f"Rank: {rang}, RegParam: {regParam}, RMSE: {rmse}")
    
    return meilleurs_params

def entrainer_modele_final(session, data_dir, params, movies_path=None, save_to_es=False):
    """
    Entraîne un modèle final avec les meilleurs paramètres sur ua.base
    et évalue sur ua.test
    """
    # Lire les données
    train_path = os.path.join(data_dir, "ua.base")
    test_path = os.path.join(data_dir, "ua.test")
    
    train_df = reader(train_path, schema_record, "\t", session)
    test_df = reader(test_path, schema_record, "\t", session)
    
    # Renommer les colonnes pour ALS
    train_df = train_df.withColumnRenamed("user_id", "userId") \
                      .withColumnRenamed("item_id", "movieId") \
                      .withColumn("rating", col("rating").cast("float"))
    test_df = test_df.withColumnRenamed("user_id", "userId") \
                     .withColumnRenamed("item_id", "movieId") \
                     .withColumn("rating", col("rating").cast("float"))
    
    # Entraîner le modèle final
    rang = params.get('rang')
    regParam = params.get('regParam')
    iterations = params.get('iterations')
    
    logger.info(f"Training final model with rank={rang}, regParam={regParam}, iterations={iterations}")
    final_model = entrainer_modele_als(train_df, rang, regParam, iterations)
    
    # Évaluer le modèle final
    final_rmse, predictions = evaluer_modele(final_model, test_df)
    logger.info(f"Final model RMSE on ua.test: {final_rmse}")
    
    # Charger les informations sur les films si disponibles
    movies_df = None
    if movies_path:
        try:
            movies_df = reader(movies_path, schema_item, "|", session)
        except Exception as e:
            logger.warning(f"Could not load movie information: {str(e)}")
    
    # Générer des recommandations pour quelques utilisateurs
    user_ids = [1, 100, 200, 300, 500]
    index_name="recommendations"
    # Create elasticsearch index for recommendations
    if save_to_es:
        create_elasticsearch_index(index_name)
    
    for user_id in user_ids:
        recs = generer_recommandations_utilisateur(final_model, user_id, 10)
        if recs:
            logger.info(f"Top 5 recommendations for user {user_id}:")
            
            # Enrichir avec les informations des films si disponibles
            if movies_df:
                enriched_recs = enrichir_recommandations(recs, movies_df)
                enriched_recs.select("userId", "movieId", "movie_title", "predicted_rating").show(5, False)
                
                # Save to Elasticsearch if specified
                if save_to_es:
                    save_recommendations_to_elasticsearch(recs, "recommendations", movies_df)
            else:
                recs.show(5, False)
                
                # Save to Elasticsearch if specified
                if save_to_es:
                    save_recommendations_to_elasticsearch(recs, "recommendations")
    
    return final_model, final_rmse

def main():
    # Initialize Spark
    spark = instanceSp()
    print(spark)
    
    try:
        # Définir le répertoire des données
        data_dir = "data"
        
        # 1. Option: Utiliser la validation croisée à 5 plis avec les données préparées
        logger.info("Performing 5-fold cross-validation...")
        
        # Paramètres par défaut pour un test rapide
        default_params = {'rang': 10, 'regParam': 0.1, 'iterations': 10}
        avg_rmse, resultats_par_pli = validation_croisee_5_plis(spark, data_dir, default_params)
        
        # 2. Option: Optimiser les hyperparamètres (cela peut prendre du temps)
        logger.info("Optimizing hyperparameters (this may take a while)...")
        meilleurs_params = optimiser_hyperparametres(spark, data_dir)
        
        # 3. Entraîner le modèle final avec les meilleurs paramètres
        logger.info("Training final model with optimized parameters...")
        movies_path = os.path.join(data_dir, "u.item")
        
        # Enable saving to Elasticsearch
        final_model, final_rmse = entrainer_modele_final(spark, data_dir, meilleurs_params, movies_path, save_to_es=True)
        
        logger.info("ALS recommendation model training completed successfully")
        logger.info("Recommendations saved to Elasticsearch")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Arrêter la session Spark
        spark.stop()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    main()