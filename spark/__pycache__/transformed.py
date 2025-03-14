from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',filename="logs.txt")
logger = logging.getLogger(__name__)

def instanceSp():
    """Initialise et retourne une session Spark"""
    instance_Spark = SparkSession.builder \
        .master("local[*]") \
        .appName('MovieLens ALS Recommender') \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    return instance_Spark

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

def entrainer_modele_als(donnees_entrainement, rang=10, regParam=0.1, iterations=10):
    """
    Crée et entraîne un modèle ALS avec les paramètres spécifiés
    """
    logger.info(f"Training ALS model with rank={rang}, regParam={regParam}, iterations={iterations}")
    
    als = ALS(
        maxIter=iterations, # comment iterations que il doit effectuer pour amelirer performance
        regParam=regParam, #limite pour pas etre plus compiliquer
        rank=rang,        #latent feature
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
            )
    
    model = als.fit(donnees_entrainement)
    logger.info("Model training completed")
    return model

def main():
    entrainer_modele_als("./../data/u.data.