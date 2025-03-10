import pandas as pd

df_links=pd.read_csv("links.csv",delimiter=",")
df_movies=pd.read_csv("movies.csv",delimiter=",")
df_ratings=pd.read_csv("ratings.csv",delimiter=",")
df_tags=pd.read_csv("./data/ml-32m/tags.csv",delimiter=",")

print(df_tags.isna().sum())
