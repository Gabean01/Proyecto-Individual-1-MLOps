from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.staticfiles import StaticFiles
from fastapi import FastAPI
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json

# Crea una instancia de la aplicación FastAPI
app = FastAPI(
    tittle = 'Modelo de recomendacion: SteamGames',
    version='1.0.0'
)

@app.get("/")
async def root():
    """

    Proyevto Individual #1 MLOps

    Version: 1.0.0

    """
    return {"Mensaje": "Proyecto realizado por Gary Bean"}



# Cargar Datos
columnstouse=['item_id','playtime_forever','user_id']
df_UserItems=pd.read_parquet("/Datasets/ArchivosPARQUET/user_items_limpio.parquet",columns=columnstouse)
df_SteamGames=pd.read_parquet("/Datasets/ArchivosPARQUET/steam_games_limpio.parquet")
df_UserReviews=pd.read_parquet("/Datasets/ArchivosPARQUET/user_reviews_limpio.parquet")
df_Recomendacion=pd.read_parquet("/Datasets/ArchivosPARQUET/recomendacion_item.parquet")

df_SteamGames=df_SteamGames.head(14000)
df_UserItems=df_UserItems.head(14000)
df_UserReviews=df_UserReviews.head(14000)
df_Recomendacion=df_Recomendacion.head(14000)

# endpoint 'developer'
@app.get('/developer')
def developer(desarrollador: str):
    # Filtrar el DataFrame de juegos por la desarrolladora especifica
    developer_df = df_SteamGames[df_SteamGames['developer'] == desarrollador]

    # Contar la cantidad de juegos y calcular el porcentaje de contenido Free por año
    developer_data = developer_df.groupby('year').agg({'item_id': 'count', 'Free to Play': 'mean'}).reset_index()
    developer_data.rename(columns={'item_id': 'Cantidad de Items', 'Free to Play': 'Contenido Free'}, inplace=True)
    developer_data['Contenido Free'] = developer_data['Contenido Free'].apply(lambda x: f"{x:.0%}")

    # Convertir el resultado a formato JSON
    result = developer_data.to_dict(orient='records')

    return result

# endpoint 'userdata'
@app.get('/userdata')
def userdata(User_id: str):
    # Filtrar el DataFrame de reseñas por el usuario especifico
    user_df = df_UserReviews[df_UserReviews['item_id'] == User_id]

    # Calcular el dinero gastado, el porcentaje de recomendacion y la cantidad de items
    money_spent = user_df['price'].sum()
    recomend_percentage = user_df['porcentaje'].mean()
    items_count = len(user_df)

    # Convertir el resultado a formato JSON
    result = {
        "Usuario": User_id,
        "Dinero gastado": f"{money_spent:.2f} USD",
        "% de recomendacion": f"{recomend_percentage:.0%}",
        "Cantidad de items": items_count
    }
    return result


# Endpoint 'UserForGenre'
@app.get('/UserForGenre')
def UserForGenre(genero: str):
    # Verificar si el género existe en los datos
    if genero not in df_UserReviews['genre'].unique():
        raise HTTPException(status_code=404, detail="El género especificado no está presente en los datos")

    # Asegurarse de que 'item_id' sea del mismo tipo de datos en ambos DataFrames
    df_UserItems['item_id'] = df_UserItems['item_id'].astype('int64')

    # Filtrar el DataFrame de reseñas por el género específico
    genre_reviews_df = df_UserReviews[df_UserReviews['genre'] == genero]

    # Fusionar los DataFrames utilizando 'item_id' como clave
    merged_df = df_UserItems.merge(genre_reviews_df[['item_id', 'year']], how='left', on='item_id')

    # Manejar valores faltantes de manera más precisa
    merged_df['playtime_forever'].fillna(0, inplace=True)

    # Realizar las operaciones de agrupación y cálculo
    total_hours_by_user_and_year = merged_df.groupby(['user_id', 'year'])['playtime_forever'].sum()
    max_user = total_hours_by_user_and_year.idxmax()

    # Obtener el usuario con más horas jugadas
    max_user_hours_by_year = total_hours_by_user_and_year.loc[max_user]

    # Generar la respuesta JSON mejorada
    result = {
        "genre": genero,
        "user_with_most_hours": {
            "user_id": max_user[0],
            "year": int(max_user[1]),
            "total_hours_played": max_user_hours_by_year
        }
    }
    return result



# endpoint 'best_developer_year'
@app.get('/best_developer_year')
def best_developer_year(año: int):
    # Filtrar el DataFrame de reseñas por el año específico y por recomendaciones positivas
    positive_reviews_df = df_UserReviews[(df_UserReviews['year'] == año) & (df_UserReviews['recommend'] == True) & (df_UserReviews['sentiment_analysis'].isin([1, 2]))]
    
    # Hacer un join entre las reseñas positivas y los juegos para obtener el nombre del juego y el desarrollador
    merged_reviews = pd.merge(positive_reviews_df, df_SteamGames[['item_id', 'name', 'developer']], on='item_id', how='left')
    
    # Contar la cantidad de juegos recomendados por desarrollador
    developer_count = merged_reviews['developer'].value_counts()
    
    # Seleccionar los top 3 de desarrolladores con más juegos recomendados
    top_3_developers = developer_count.head(3)
    
    # Convertir el resultado a formato JSON
    result = [{"Puesto {}".format(i+1): developer} for i, (developer, _) in enumerate(top_3_developers.items())]
    return result



# endpoint 'developer_reviews_analysis'
@app.get('/developer_reviews_analysis')
def developer_reviews_analysis(desarrolladora: str):
    # Filtrar las reseñas por desarrolladora específica en base a la tabla steam_games
    filtered_reviews = df_UserReviews[df_UserReviews['name'].isin(df_SteamGames[df_SteamGames['developer'] == desarrolladora]['name'])]
    
    # Calcular el análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].value_counts()
    
    # Obtener el nombre de la desarrolladora
    developer_name = desarrolladora
    
    # Crear el diccionario de resultados
    result = {developer_name: {
        'Negative': sentiment_counts.get(0, 0),
        'Neutral': sentiment_counts.get(1, 0),
        'Positive': sentiment_counts.get(2, 0)
    }}
    
    return result


# Endpoint para obtener recomendaciones de juegos similares
# Crear un vectorizador de texto
vectorizador_texto = CountVectorizer()
vectores_texto = vectorizador_texto.fit_transform(df_Recomendacion['specs']).toarray()

# Calcular la similitud del coseno entre vectores
similitud_cos = cosine_similarity(vectores_texto)

@app.get('/recomendacion_juego/{id}')
def recomendacion_juego(id: int):
    if id not in df_Recomendacion['item_id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    
    indice_producto = df_Recomendacion[df_Recomendacion["item_id"] == id].index[0]
    distancias = similitud_cos[indice_producto]
    lista_juegos_similares = sorted(list(enumerate(distancias)), reverse=True, key=lambda x: x[1])[1:6]
    juegos_recomendados = [df_Recomendacion.iloc[i[0]]['name'] for i in lista_juegos_similares]
    
    return {'juegos_recomendados': juegos_recomendados}