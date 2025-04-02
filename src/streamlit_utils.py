import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor

def fixNaN(source_df):
    df = source_df.copy()
    # Fix NaN
    cond = (df['Nom du compteur'] == 'Face au 48 quai de la marne NE-SO') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100047542-103047542'
    df.loc[cond, 'Identifiant du site de comptage'] = 100047542
    df.loc[cond, 'Nom du site de comptage'] = 'Face au 48 quai de la marne'
    df.loc[cond, "Date d'installation du site de comptage"] = '2018-11-29'
    df.loc[cond, 'Coordonnées géographiques'] = '48.89128,2.38606'

    cond = (df['Nom du compteur'] == 'Face au 48 quai de la marne SO-NE') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100047542-104047542'
    df.loc[cond, 'Identifiant du site de comptage'] = 100047542
    df.loc[cond, 'Nom du site de comptage'] = 'Face au 48 quai de la marne'
    df.loc[cond, "Date d'installation du site de comptage"] = '2018-11-29'
    df.loc[cond, 'Coordonnées géographiques'] = '48.89128,2.38606'

    cond = (df['Nom du compteur'] == 'Quai des Tuileries NO-SE') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100056035-353266462'
    df.loc[cond, 'Identifiant du site de comptage'] = 100056035
    df.loc[cond, 'Nom du site de comptage'] = 'Quai des Tuileries'
    df.loc[cond, "Date d'installation du site de comptage"] = '2021-05-18'
    df.loc[cond, 'Coordonnées géographiques'] = '48.8635,2.32239'

    cond = (df['Nom du compteur'] == 'Quai des Tuileries SE-NO') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100056035-353266460'
    df.loc[cond, 'Identifiant du site de comptage'] = 100056035
    df.loc[cond, 'Nom du site de comptage'] = 'Quai des Tuileries'
    df.loc[cond, "Date d'installation du site de comptage"] = '2021-05-18'
    df.loc[cond, 'Coordonnées géographiques'] = '48.8635,2.32239'

    cond = (df['Nom du compteur'] == 'Pont des Invalides N-S') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100056223-101056223'
    df.loc[cond, 'Identifiant du site de comptage'] = 100056223
    df.loc[cond, 'Nom du site de comptage'] = 'Pont des Invalides'
    df.loc[cond, "Date d'installation du site de comptage"] = '2019-11-07'
    df.loc[cond, 'Coordonnées géographiques'] = '48.86281,2.31037'

    cond = (df['Nom du compteur'] == '10 avenue de la Grande Armée SE-NO') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100044494-353504987'
    df.loc[cond, 'Identifiant du site de comptage'] = 100044494
    df.loc[cond, 'Nom du site de comptage'] = '10 avenue de la Grande Armée'
    df.loc[cond, "Date d'installation du site de comptage"] = '2018-07-27'
    df.loc[cond, 'Coordonnées géographiques'] = '48.8748,2.2924'

    cond = (df['Nom du compteur'] == '27 quai de la Tournelle NO-SE') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100056336-104056336'
    df.loc[cond, 'Identifiant du site de comptage'] = 100056336
    df.loc[cond, 'Nom du site de comptage'] = '27 quai de la Tournelle'
    df.loc[cond, "Date d'installation du site de comptage"] = '2019-11-14'
    df.loc[cond, 'Coordonnées géographiques'] = '48.85013,2.35423'

    cond = (df['Nom du compteur'] == '27 quai de la Tournelle SE-NO') & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = '100056336-103056336'
    df.loc[cond, 'Identifiant du site de comptage'] = 100056336
    df.loc[cond, 'Nom du site de comptage'] = '27 quai de la Tournelle'
    df.loc[cond, "Date d'installation du site de comptage"] = '2019-11-14'
    df.loc[cond, 'Coordonnées géographiques'] = '48.85013,2.35423'

    # Fusion des compteurs doublons identifiés pendant l'exploration
    df = df.replace({
        'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO': 'Face au 48 quai de la marne NE-SO',
        'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE': 'Face au 48 quai de la marne SO-NE',
        'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos E-O': 'Totem 64 Rue de Rivoli E-O',
        'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos O-E': 'Totem 64 Rue de Rivoli O-E',
        'Quai des Tuileries Quai des Tuileries Vélos NO-SE': 'Quai des Tuileries NO-SE',
        'Quai des Tuileries Quai des Tuileries Vélos SE-NO': 'Quai des Tuileries SE-NO',
        'Pont des Invalides (couloir bus)': 'Pont des Invalides',
        '69 Boulevard Ornano (temporaire)': '69 Boulevard Ornano',
        '30 rue Saint Jacques (temporaire)': '30 rue Saint Jacques',
        '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE': '27 quai de la Tournelle NO-SE',
        '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO': '27 quai de la Tournelle SE-NO',
        'Pont des Invalides (couloir bus) N-S': 'Pont des Invalides N-S',
    })

    # Drop des 2 observations des compteurs dont la direction n'est pas claire
    df = df.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]'].index)
    df = df.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]'].index)
    df = df.drop(columns = ["Identifiant technique compteur", "ID Photos", "test_lien_vers_photos_du_site_de_comptage_", "id_photo_1", "url_sites", "type_dimage", "mois_annee_comptage", "Lien vers photo du site de comptage"])
    df['Date et heure de comptage'] = pd.to_datetime(df['Date et heure de comptage'], utc=True)
    df['Date et heure de comptage'] = df['Date et heure de comptage'].dt.tz_convert("Europe/Paris")
    df['year'] = df['Date et heure de comptage'].dt.year
    df['month'] = df['Date et heure de comptage'].dt.month
    df['day'] = df['Date et heure de comptage'].dt.day
    df["date"] = df["Date et heure de comptage"].dt.date
    df['weekday'] = df['Date et heure de comptage'].dt.weekday
    df['hour'] = df['Date et heure de comptage'].dt.hour
    df['mois_annee_comptage'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    df = df.loc[df.year >= 2023]
    return df

@st.cache_data
def get_lieux_compteurs_df(source_df):
    df = source_df[['Nom du site de comptage', 'Date et heure de comptage', 'Comptage horaire']]
    df = df.groupby(by=['Nom du site de comptage', 'Date et heure de comptage'], as_index=False)['Comptage horaire'].sum()
    return df

def plotly_map(df):
    df_geo = df["Coordonnées géographiques"].str.split(',', expand=True).rename(columns={0: "latitude", 1: "longitude"})
    df_geo = pd.concat([df, df_geo], axis=1)
    df_group = df_geo.groupby("Nom du compteur", as_index=False).agg({
        "Comptage horaire": "sum",
        "latitude": "first",
        "longitude": "first"
    })
    return px.scatter_map(
        df_group,
        lat="latitude",
        lon="longitude",
        size="Comptage horaire",
        color="Comptage horaire",
        size_max=30,
        center={"lat": 48.86, "lon": 2.335},
        zoom=11,
        hover_name="Nom du compteur",
        width=800,
        height=750)

@st.cache_data
def load_classification_data():
    df = pd.read_csv(r"C:\Users\kevin_2crsr2j\Desktop\VELO DST\dataset_classification.csv", index_col=0)
    
    df['Comptage horaire'] = df["Comptage horaire"].replace({
        "0-3": "[00] 0-3", "4-9": "[01] 4-9", "10-18": "[02] 10-18",
        "19-31": "[03] 19-31", "32-46": "[04] 32-46", "47-64": "[05] 47-64",
        "65-86": "[06] 65-86", "87-115": "[07] 87-115", "116-155": "[08] 116-155",
        "156-230": "[09] 156-230", "231-450": "[10] 231-450", "451+": "[11] 451+",
    })

    return df

@st.cache_data
def load_regression_data():
    df_reg = pd.read_csv(r"C:\Users\kevin_2crsr2j\Desktop\VELO DST\lieu-compteur-one-hot-encoded.csv", index_col=0)
    return df_reg

@st.cache_resource
def train_classification_model(X, y, params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

@st.cache_resource
def train_regression_model(X, y, params):
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return model
