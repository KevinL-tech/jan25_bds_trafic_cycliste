from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from src.streamlit_utils import load_classification_data, load_regression_data, plotly_map, fixNaN, get_lieux_compteurs_df, train_classification_model, train_regression_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px

# Configuration des chemins pour accéder aux modèles pré-entraînés
MODELS_DIR = Path("models")

@st.cache_data
def load_raw_data():
    df_2023 = pd.read_csv('data/raw/2023-comptage-velo-donnees-compteurs.csv', sep=';')
    df_2024 = pd.read_csv('data/raw/2024-comptage-velo-donnees-compteurs.csv', sep=';') 
    df = pd.concat([df_2023, df_2024], axis=0)
    df = fixNaN(df)
    return df

# Charger les données
df = pd.read_csv("data/processed/lieu-compteur-one-hot-encoded.csv", index_col=0)

st.image("streamlit_assets/banniere6.jpeg", use_container_width=True)

#Titres
st.title("**Projet Data Science - Trafic Cycliste à Paris**")
st.subheader("_Données de Janvier 2023 à Février 2025_")

st.sidebar.title("Sommaire")
pages=["Présentation du Projet", "Présentation du dataset", "Préprocessing", "Visualisation des données", "Modélisation", "Interprétation et résultats", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("""
_Projet réalisé par:_
- _Alexandre COURROUX_
- _Kévin LAKHDAR_
- _Ketsia PEDRO_
- _Eliah REBSTOCK_ 

_Bootcamp Machine Learning Engineer Janvier 2025_
""")


## Présentation du projet
if page == pages[0]:
    st.header("Présentation du projet", divider=True)

    st.header("I. Contexte")

    st.markdown("""
Face à l'essor du vélo comme mode de transport durable, la Ville de Paris a mis en place, depuis plusieurs années, un réseau de compteurs à vélo permanents pour mesurer l'évolution de la pratique cycliste dans la capitale.

Ces capteurs, installés sur les axes clés de la ville, collectent en continu des données sur le flux des cyclistes.
    """)

    st.image("streamlit_assets/comptagevélo.jpeg", use_container_width=True)

    st.markdown("""
Ce projet s'inscrit dans une démarche de transition vers une mobilité plus verte et une volonté d'adapter les infrastructures urbaines aux besoins réels, tel que proposé dans le plan vélo 2021-2026 d'aménagement de pistes cyclables de la mairie de Paris.

L'enjeu est de transformer ces données brutes en insights exploitables, permettant d'éclairer les décisions publiques de manière objective et data-driven.
    """)

    st.divider()

    st.header("II. Objectifs")
    st.markdown("Ce projet vise à développer un **outil prédictif du trafic cycliste à Paris**, en exploitant les données historiques des compteurs vélo.")

    st.subheader("Objectifs principaux")
    st.markdown("""
- Identifier les **tendances d’usage** (heures de pointe, zones saturées, variations saisonnières).
- Générer des **visualisations claires** (cartes thermiques, graphiques temporels).
- Aider à la **prise de décision** sur les aménagements à prioriser.
""")

    st.subheader("Bénéfices pour la Mairie de Paris :")
    st.markdown("""
- Prioriser les **aménagements ciblés** (pistes élargies, carrefours sécurisés, nouveaux itinéraires).
- Évaluer l’**impact des politiques existantes**.
- **Optimiser le réseau cyclable** à long terme.
""")

    st.subheader("Ambition finale")
    st.markdown("> Réduire les **conflits d’usage**, améliorer la **sécurité**, et encourager la pratique du vélo grâce à une **planification data-driven**, combinant **rétrospective** et **prédiction** pour une mobilité plus fluide et résiliente.")








## Présentation du dataset
if page == pages[1]: 
    st.header("Présentation du dataset", divider=True)
    st.header("1. Sources des données")
    st.image("streamlit_assets/opendata2.png", use_container_width=True)
    st.markdown("""
Utilisation des jeux de données ouverts proposés par la Ville de Paris via [opendata.paris.fr](https://opendata.paris.fr) :

  - le jeu de données [Comptage vélo - Données compteurs](https://opendata.paris.fr/explore/dataset/comptage-velo-donnees-compteurs/information) pour les données de 2024-2025.
  - le jeu de données [Comptage vélo - Historique - Données Compteurs et Sites de comptage](https://opendata.paris.fr/explore/dataset/comptage-velo-historique-donnees-compteurs/information) pour les données de 2023.
                 
Les données sont publiées sous la licence Open Database License (ODbL), qui autorise la réutilisation, l’adaptation et la création de travaux dérivés à partir de ces jeux de données, à condition d’en citer la source.
""") 

    st.divider()

    st.header("2. Période ")
    st.markdown("""
Les données sont mises à jour quotidiennement. 

Nous avons récupéré toutes les données du 1er janvier 2023 au 28 février 2025 (26 mois).                
""") 

    st.divider()

    st.header("3. Contenu des jeux de données  ")
    st.markdown("""
Les jeux de données recensent les comptages horaires des passages de vélos effectués par environ une centaine de compteurs répartis dans la ville de Paris.
                  
Chaque lieu de comptage est généralement équipé de deux compteurs, face à face, positionnés pour mesurer le trafic dans chaque direction d’une même rue. 
                 
Au total, pour la période 2023-2024, le jeu de données contient environ 1,8 million de lignes réparties sur 16 variables.
""")

    st.divider()

    st.header("4. Structure des données")
    st.markdown("""
Chaque ligne du dataset correspond au nombre de vélos enregistrés pendant une heure par un compteur donné.  

Les données incluent, en plus du comptage horaire, plusieurs métadonnées associées au compteur ou au site de comptage, telles que :
- Le nom et l’identifiant du compteur  
- Le nom du lieu de comptage (en général nº + rue)
- La date d’installation du compteur
- Les coordonnées géographiques  
- Éventuellement, des liens vers des photos du site
""")

    st.divider()
    st.header("5. Nettoyage et sélection des variables  ")

    st.markdown("""
Afin de simplifier et d’optimiser l’analyse, nous avons supprimé les variables non pertinentes pour l'entraînement du modèle, en particulier les champs techniques ou visuels comme les liens vers les photos, les identifiants internes ou le type d’image.  

Voici un extrait de notre dataset avec les variables que nous avons décidé de conserver :
""")

    st.dataframe(load_raw_data().sample(5))

    st.divider()

    st.header("6. Objectif d’analyse et variable cible  ")
    st.markdown("""
L’objectif de notre étude est de prédire le nombre de vélos comptés pendant une heure sur un compteur donné.  
La variable cible de notre modèle est donc le comptage horaire, un indicateur clé pour analyser l'évolution de la circulation cycliste dans Paris.
""")

    st.divider()
    st.header("7. Forces et limites du dataset")

    st.markdown("""
Le dataset se distingue par sa précision horaire et sa couverture géographique dense, ce qui permet d’identifier des tendances temporelles comme les variations quotidiennes ou saisonnières du trafic cycliste.  

Cependant, il ne contient pas d’informations contextuelles telles que :
- La météo 
- La présence d’événements particuliers (manifestations, grèves, festivals) 
- Ou des données sociodémographiques comme la densité de population par zone 

Cette absence limite la profondeur des analyses prédictives que l’on peut mener.
""")

## Préprocessing
if page == pages[2]: 
    st.header("Préprocessing des données", divider=True)
    st.header("1. Suppression des NaN")

    st.markdown("""
    Certaines variables de métadonnées des compteurs ("Identifiant du compteur", "Coordonnées géographiques", ...) ont des valeurs NaN (environ 3.4% sur le dataset).

    Plusieurs compteurs du dataset correspondent en réalité à un même emplacement, ce qui a permis de réduire les NaN en les renommant et fusionnant. 

    Les derniers NaN provenaient de deux compteurs atypiques, finalement supprimés pour obtenir un dataset complet et sans valeurs manquantes.
    """) 
        
    st.header("2. Conversion Date au format datetime") 
                    
    st.markdown("""
    Variable "Date et heure de comptage" convertie au format datetime de Pandas, en utilisant le fuseau horaire Europe/Paris afin de correctement capturer les tendances journalières.
    """) 
                    
    st.code("""
    # Convertir la colonne en datetime (avec gestion du fuseau horaire)
    df['Date et heure de comptage'] = pd.to_datetime(df['Date et heure de comptage'], utc=True)
    df['Date et heure de comptage'] = df['Date et heure de comptage'].dt.tz_convert("Europe/Paris")
    """, language="python")

    st.header("3. Ajout de variables")
    st.markdown("""                
    La variable "Date et heure de comptage" décomposée en variables "année", 
    "mois", "jour", "jour de la semaine" et "heure" afin de faciliter la data visualisation et voir si 
    certaines de ces variables étaient corrélés à notre variable cible.
    """)
    
    st.code("""
    df["Jour"] = df["Date et heure de comptage"].dt.date
    df["Mois"] = df["Date et heure de comptage"].dt.month
    df["Année"] = df["Date et heure de comptage"].dt.year
    df["Heure"] = df["Date et heure de comptage"].dt.hour
    """, language="python")

    st.markdown('Ajout des variables catégorielles binaires "Week-end", "Jour fériés" et "Vacances scolaires" afin de mesurer si les jours non travaillés ont un impact sur la pratique cyclable.') 

    st.header("4. Normalisation des données")

    st.markdown("""
    Nous avons appliqué deux types de **normalisation** sur les colonnes temporelles et contextuelles, notamment pour réduire l'impact des valeurs extrêmes de Comptage horaire sur les prédictions de notre modèle, la variable Comptage horaire ne suivant pas une loi normale :

    1. **Standardisation** : centre les données autour de 0 avec une variance de 1.
    2. **Min-Max Scaling** : transforme les valeurs dans une plage définie, ici entre 0 et 1.
    """)

    st.subheader("🔹 Standardisation")

    st.code("""
    from sklearn.preprocessing import StandardScaler

col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]

scaler = StandardScaler()
df[col_norm] = scaler.fit_transform(df[col_norm])
    """, language="python")

    st.subheader("🔹 Normalisation Min-Max")

    st.code("""
from sklearn.preprocessing import MinMaxScaler

col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]

scaler = MinMaxScaler(feature_range=(0, 1))
df[col_norm] = scaler.fit_transform(df[col_norm])
    """, language="python")

    st.markdown("""
    Ces transformations permettent de préparer les données pour les modèles sensibles à l’échelle des variables (régressions, KNN, etc.).
    """)

    st.header("Extrait du Dataframe après pré-processing")
    st.dataframe(df.head(10))




## Visualisation des données                
if page == pages[3]: 
    raw_data = load_raw_data()

    st.header("Visualisation des données", divider=True)
    st.header("I. Cartographie")
    st.markdown("Carte de la ville de Paris représentant les positions des différents compteurs du dataset. La taille de chaque point correspond au comptage horaire total sur la période 2023-2024.")

    st.plotly_chart(plotly_map(load_raw_data()))

    st.markdown("""
    Les compteurs sont répartis sur les axes principaux :
            
    - Sud-Ouest / Nord-Est (avenue Denfert-Rochereau et boulevard Sébastopol) 
    - Est-Ouest (avenue de la Grande Armée et avenue des Champs-Élysées). 
    - Les quais de la Seine ainsi que le boulevard périphérique Sud (le long de la voie de tram T3a) sont aussi couverts. 
                
    Le boulevard périphérique nord et les 17 et 18e arrondissements n'ont pas de compteurs.
    
    Les compteurs "centraux" ont plus de passages que ceux en périphérie de Paris : il y a donc une corrélation entre la localisation du compteur et le comptage horaire.""")

    st.header("II. Évolution temporelle")

    st.subheader("""a. Évolution globale du trafic""")
    fig = plt.figure(figsize=(12, 5))
    comptage_quotidien = raw_data.groupby("date")["Comptage horaire"].sum()
    plt.plot(comptage_quotidien.index.astype(str), comptage_quotidien.values, linestyle = "-", color = "orange")

    plt.xlabel("Jour")
    plt.ylabel("Nombre de vélos")
    plt.title("Évolution du comptage quotidien des vélos")
    plt.xticks(ticks = range(0, len(comptage_quotidien), 15), labels = comptage_quotidien.index[::15].astype(str), rotation = 45, fontsize = 8)

    plt.grid(axis = "y", linestyle = "--", alpha = 0.7)
    st.pyplot(fig)

    st.markdown("""
On peut observer ici certaines tendances notamment des diminutions significatives
au mois d'Août et Décembre qui correspondent respectivement à une saison chaude
pendant les vacances scolaires (voyages, etc.) et à Noël (saison plus froide et familiale).
Il semble également y avoir une reprise au mois de Septembre montrant la reprise du travail 
et la rentrée pour les étudiants.
""")

    st.subheader("""b. Saisonnalité du trafic""")
    
    fig = plt.figure()
    sns.barplot(df, x='Mois', y='Comptage horaire', errorbar=None)
    plt.xlabel("Mois")
    plt.xticks(rotation=45)
    plt.title("Comptage horaire moyen en fonction du mois")
    st.pyplot(fig)

    st.markdown("""
    On constate une baisse du comptage en **hiver** (janvier et décembre) et en **été** (au mois d'août).

    Cela est peut-être dû aux **vacances**, à certains **événements** (JO de Paris en août en 2024) et à la **météo** (il fait plus froid en hiver, ce qui n'encourage pas la pratique cycliste).""")

    st.subheader("""c. Comportement selon les jours""")
    
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(df, x='Jour_semaine', y='Comptage horaire', errorbar=None)
    plt.xlabel("Jour du mois")
    plt.title("Comptage horaire moyen en fonction du jour de la semaine")
    st.pyplot(fig)

    st.markdown("""
    On constate également plus de **comptages** du **lundi au vendredi**, ce qui correspond aux **trajets domicile-travail**.

    En moyenne, il y a environ **50% de vélos en plus** en **semaine** par rapport au **week-end**.
    """)


    st.subheader("""d. Evolution du trafic au fil des heures""")

    st.markdown("""À gauche : **jours de la semaine** (lundi à vendredi) — À droite : **week-end** (samedi et dimanche)""")

    # Filtrage
    df_semaine = df[df['Jour_semaine'].isin([1, 2, 3, 4, 5])]
    df_weekend = df[df['Jour_semaine'].isin([6, 7])]

    # Détermination du même axe Y pour les deux graphiques
    y_max = max(df['Comptage horaire'].max(), 1)  # max global pour fixer l'échelle

    # Création des deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    # Graphique semaine
    sns.barplot(ax=axes[0], data=df_semaine, x='Heure', y='Comptage horaire', errorbar=None)
    axes[0].set_title("Comptage horaire moyen (lundi à vendredi)")
    axes[0].set_xlabel("Heure de la journée")
    axes[0].set_ylabel("Comptage horaire")
    axes[0].set_ylim(0, 400)

    # Graphique week-end
    sns.barplot(ax=axes[1], data=df_weekend, x='Heure', y='Comptage horaire', errorbar=None)
    axes[1].set_title("Comptage horaire moyen (week-end)")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("")  # on peut laisser vide pour alléger visuellement
    axes[1].set_ylim(0, 400)

    st.pyplot(fig)

    st.markdown("""
    **Forte augmentation du trafic** aux **heures de pointe** (8h–9h / 18h–19h) en semaine.

    **Volume de passages** relativement **régulier** entre **11h et 20h** le **week-end**.
    """)

    st.header("III. Distribution de la variable cible")
    st.subheader('Boxplot global de la variable comptage horaire')

    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(raw_data, x='Comptage horaire')
    st.pyplot(fig)

    st.markdown('On remarque la présence de nombreuses valeurs extrêmes, au sens statistique. 75% des valeurs sont inférieures à 95 comptages par heure et 25% des valeurs sont inférieures à 11.')

    st.subheader('Histogramme de la variable comptage horaire')

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(raw_data, x='Comptage horaire', kde=True)
    plt.xlim(0, 2000)
    st.pyplot(fig)

    st.markdown("L'histogramme confirme le nombre élevé de valeurs inférieures à 100.")

    st.subheader("QQ-Plot de la variable comptage horaire")
    import statsmodels.api as sm

    fig = sm.qqplot(df["Comptage horaire"], line = "r")
    plt.title("QQ-Plot de Comptage horaire")
    st.pyplot(fig)

    st.markdown("""
    On peut se rendre compte que la distribution de la variable n'est pas normale.
    Ceci peut s'expliquer par le fait que les données soient concentrées vers 0.
    """)

    st.subheader('Boxplot de la variable comptage horaire en fonction du lieu de comptage') 

    agg_data = get_lieux_compteurs_df(raw_data)
    site = st.selectbox("Nom du site de comptage", list(agg_data['Nom du site de comptage'].unique()))
    fig = plt.figure(figsize=(10, 5))
    sns.boxplot(agg_data.loc[agg_data['Nom du site de comptage'] == site], x='Comptage horaire')
    plt.title(site)
    st.pyplot(fig)

    
    st.header("IV. Corrélation entre les variables")
    
    st.subheader("Matrice de corrélation entre les variables")

    st.image("streamlit_assets/matrice.jpeg", use_container_width=True)

    st.markdown("""
    * Le **comptage horaire** est légèrement corrélé au **nom du compteur** et à **l'heure de la journée**.
    * Corrélation forte entre les variables **jour_semaine** et **week-end** (variables redondantes).
    * Forte corrélation entre **"date et heure de comptage"**, **année** et **mois**, ce qui semble logique, l'année et le mois étant dérivés de la date.
    """)


# Modélisation   
if page == pages[4]: 
    st.header("Modélisation", divider=True)
    problem_type = st.segmented_control("Type de problème", ["Classification", "Régression"])
    
    if problem_type == 'Classification':
        # Chargement des données
        class_df = load_classification_data()
        X = class_df.drop(columns=["Comptage horaire"])
        col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]
        
        # Encodage des features
        encoder = OneHotEncoder(sparse_output=False, dtype=int) 
        array = encoder.fit_transform(X[col_norm])
        encoded_df_clean = pd.DataFrame(array, columns=encoder.get_feature_names_out(col_norm))
        encoded_df_clean.index = X.index
        X_clean = pd.concat([X.drop(columns=col_norm), encoded_df_clean], axis=1)

        # Encodage variable cible
        label_enc = LabelEncoder()
        y = class_df["Comptage horaire"]  
        y = label_enc.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

        st.header("Analyse de Classification")
        st.write("**Modèle choisi** : `RandomForestClassifier`")

        with st.expander("Hyperparamètres de Classification", expanded=True):
            n_estimators = st.slider("n_estimators", 50, 200, 200, 50)
            max_depth = st.slider("max_depth", 10, 100, 70, 10)
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'criterion': 'gini', 'min_samples_split':15, 'min_samples_leaf':2, 'max_features':'sqrt'}
            
            # On génère un nom de fichier unique qui est basé sur les hyperparamètres
            model_filename = MODELS_DIR / f"rf_classifier_{n_estimators}_{max_depth}.pkl"
            
            # Ici on vérifie si le modèle est déjà entraîné
            if model_filename.exists():
                st.info("Chargement du modèle pré-entraîné...")
                model = joblib.load(model_filename)
                st.success("Modèle chargé.")
            else:
                st.info("Entraînement du modèle... (peut prendre quelques minutes)")
                model = train_classification_model(X_train, y_train, params)
                joblib.dump(model, model_filename)
                st.success("Modèle entraîné et sauvegardé pour une utilisation future !")
        
        # Prédictions et évaluation
        y_pred = model.predict(X_test)

        st.subheader("Performance du Modèle")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=label_enc.classes_, output_dict=True)).transpose())
    
        fig = px.imshow(confusion_matrix(y_test, y_pred),
                        labels=dict(x="Prédit", y="Réel", color="Count"),
                        x=label_enc.classes_, y=label_enc.classes_,
                        color_continuous_scale='Blues')
        fig.update_layout(title="Matrice de confusion")
        st.plotly_chart(fig, use_container_width=True)
    
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
    
        fig = px.bar(feature_importance, x='importance', y='feature', 
                     title='Top 10 des caractéristiques importantes')
        st.plotly_chart(fig, use_container_width=True)

    if problem_type == 'Régression':
        # Chargement des données
        reg_df = load_regression_data()
        X = reg_df.drop(columns=["Comptage horaire"])
        col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]
        
        # Encodage des features
        encoder = OneHotEncoder(sparse_output=False, dtype=int) 
        array = encoder.fit_transform(X[col_norm])
        encoded_df_clean = pd.DataFrame(array, columns=encoder.get_feature_names_out(col_norm))
        encoded_df_clean.index = X.index
        X_clean = pd.concat([X.drop(columns=col_norm), encoded_df_clean], axis=1)

        # Transformation logarithmique de la variable cible pour éviter les valeurs aberrantes
        y = np.log1p(reg_df["Comptage horaire"])  # log(1+x) pour gérer les zéros
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

        st.header("Analyse de Régression")
        st.write("""
        **Modèle choisi:** `HistGradientBoostingRegressor`
        
        **Note:** Une transformation logarithmique a été appliquée à la variable cible pour:
        - Réduire l'impact des valeurs extrêmes
        - Éviter les prédictions négatives
        - Normaliser la distribution des données
        """)
    
        with st.expander("Hyperparamètres de Régression", expanded=True):
            learning_rate = st.slider("Taux d'apprentissage", 0.1, 0.5, 0.1, 0.1)
            max_iter = st.slider("Nombre d'itérations", 50, 500, 100, 50)
            params = {'learning_rate': learning_rate, 'max_iter': max_iter}

            model_filename = MODELS_DIR / f"hgb_regressor_{learning_rate}_{max_iter}.pkl"

            # Ici on vérifie si le modèle est déjà entraîné
            if model_filename.exists():
                st.info("Chargement du modèle pré-entraîné...")
                model = joblib.load(model_filename)
                st.success("Modèle chargé.")
            else:
                st.info("Entraînement du modèle... (peut prendre quelques minutes)")
                model = train_regression_model(X_train, y_train, params)
                joblib.dump(model, model_filename)
                st.success("Modèle entraîné et sauvegardé pour une utilisation future !")

        y_pred = model.predict(X_test)
        
        # Conversion inverse des prédictions (expm1 pour inverser log1p)
        y_test_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
    
        st.subheader("Performance du Modèle")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test_exp, y_pred_exp)):.2f}")
        col2.metric("R²", f"{r2_score(y_test_exp, y_pred_exp):.2f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test_exp, y_pred_exp):.2f}")
    
        fig = px.scatter(x=y_test_exp, y=y_pred_exp, 
                         labels={'x': 'Valeurs Réelles', 'y': 'Prédictions'},
                         title='Valeurs Réelles vs Prédictions (échelle originale)')
        fig.add_shape(type='line', line=dict(dash='dash'),
                      x0=min(y_test_exp), y0=min(y_test_exp),
                      x1=max(y_test_exp), y1=max(y_test_exp))
        st.plotly_chart(fig, use_container_width=True)
    
        residuals = y_test_exp - y_pred_exp
        fig = px.histogram(residuals, nbins=50, 
                           title='Distribution des Résidus',
                           labels={'value': 'Résidu'})
        st.plotly_chart(fig, use_container_width=True)


## Interprétation et résultats
if page == pages[5]: 
    st.header("Interprétation et résultats", divider=True)
    st.header("Analyse comparative des erreurs")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification")
        st.write("Meilleures métriques obtenues (`n_estimators=200, max_depth=100`) :")
        st.code("""
        - Accuracy : 0.55
        - Précision moyenne : 0.56
        - Rappel moyen : 0.57
        - f1-score moyen : 0.56
        """)

    with col2:
        st.subheader("Régression")
        st.write("Meilleures métriques obtenues (`learning_rate=0.1, max_iter=5000`):")
        st.code("""
        - RMSE : 21.99
        - R² : 0.96
        """)

    st.header("Interprétabilité")
    st.subheader("Importances des features pour le modèle HistGradientBoostingRegressor")

    st.image("streamlit_assets/feature importances.png", use_container_width=True)

    st.markdown("""
Nous pouvons constater ici que la caractéristique la plus importante pour le modèle est
l’heure et plus largement tout ce qui se rapporte à la date (heure, mois, jour de la semaine
et année). Cela reste cohérent avec l’objectif fixé qui est de prédire le nombre de vélos sur
un compteur à une heure précise.
""")

    st.header("Conclusions")

    st.subheader("Classification")
    st.image("streamlit_assets/erreurs_classes.png", use_container_width=True)
    st.write("""
    - Les classes extrêmes (faible et haut trafic) sont les mieux prédites.
    - Les classes intermédiaires sont moins biens prédites à cause d'erreurs de classement entre classes voisines.
    - L'heure et le jour de la semaine sont les facteurs les plus déterminants.
    """)

    st.subheader('Régression')
    st.image("streamlit_assets/scores_r2.png", use_container_width=True)
    st.write("""
    - Bonnes performances globales avec un R² de 0.96.
    - Les erreurs augmentent lors des pics de trafic exceptionnels.
    - Le modèle capture bien les variations saisonnières horaires.
    """)

## Conclusion
if page == pages[6]: 
    st.header("Synthèse du Projet", divider=True)

    st.markdown("""
    Ce projet de data science vise à **prédire le trafic cycliste à Paris** à l'aide des données ouverte de comptage horaire. 
    L'objectif principal était de développer un modèle capable d'estimer avec précision le nombre de vélos circulant 
    sur les axes cyclables parisiens, afin d'aider la ville dans sa politique d'aménagement urbain.
    
    **Principales réalisations :**

    - Collecte et traitement de **1,8 million d'observations** (2023-2025)
    - Analyse exploratoire approfondie des tendances du trafic cycliste.
    - Développement de deux approches :
        - **Modélisation par régression** pour une prédiction précise du nombre de vélos.
        - **Classification** pour catégoriser l'intensité du trafic.
    - Création d'une application interactive pour visualiser les résultats.
    """)

    st.divider()

    st.header("Principaux Enseignements")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Insights Clés")
        st.markdown("""
        - L'**heure** et le **jour de la semaine** sont les facteurs les plus prédictifs.
        - Forte variation entre heures de pointe et périodes creuses.
        - Impact visible des **vacances scolaires** et jours fériés.
        - Différences marquées selon les **localisations géographiques**.
        """)
    
    with col2:
        st.subheader("⚙️ Performance des Modèles")
        st.markdown("""
        - **Régression** (meilleure performance):
            - Précision moyenne : **±10 vélos/heure**
            - Capacité à capturer les tendances saisonnières
        - **Classification** :
            - Précision globale de **56%**
            - Bonne détection des pics et creux de trafic
        """)

    st.divider()

    st.header("Applications Concrètes")
    st.markdown("""
    🔹 **Pour la Mairie de Paris :**

    - Optimisation des **aménagements cyclables**
    - Meilleure gestion des **flux aux heures de pointe**
    - Aide à la décision pour les **investissements infrastructurels**
    
    🔹 **Pour les Citoyens :**

    - Application potentielle pour **éviter les axes saturés**
    - Visibilité sur les **tendances du trafic cycliste**
    """)

    st.divider()

    st.header("Perspectives d'Améliorations")
    st.markdown("""
    🚀 **Améliorations Techniques:**

    - Intégration de données **météorologiques**
    - Ajout d'informations sur les **événements locaux**
    - Utilisation de techniques avancées (deep learning, modèles séquentiels)
    
    🌍 **Extensions Possibles:**

    - Prédiction à l'échelle de la **semaine/mois**
    - Analyse comparative entre **différentes villes**
    - Système de recommandation d'itinéraires cyclables
    """)

    st.divider()

    st.markdown("""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px;">
    <h3 style="color:#1e88e5;">En résumé</h3>
    <p>Ce projet démontre la valeur des données de mobilité pour la gestion urbaine. 
    Les résultats obtenus ouvrent des perspectives intéressantes pour une ville comme Paris 
    qui souhaite développer les mobilités douces tout en optimisant ses infrastructures existantes.</p>
    </div>
    """, unsafe_allow_html=True)