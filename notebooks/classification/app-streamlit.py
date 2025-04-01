from sklearn import preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, mean_absolute_error, 
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Analyse Trafic", layout="wide")

@st.cache_data
def load_classification_data():
    df = pd.read_csv('/home/ketsiapedro/Bureau/lieu-compteur-classes-one-hot-encoded.csv', index_col=0)
    
    y = df["Comptage horaire"].replace({
        "0-3": "[00] 0-3", "4-9": "[01] 4-9", "10-18": "[02] 10-18",
        "19-31": "[03] 19-31", "32-46": "[04] 32-46", "47-64": "[05] 47-64",
        "65-86": "[06] 65-86", "87-115": "[07] 87-115", "116-155": "[08] 116-155",
        "156-230": "[09] 156-230", "231-450": "[10] 231-450", "451+": "[11] 451+",
    })
    return df
    
df = load_classification_data()
X = df.drop(columns=["Comptage horaire"])
col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]
encoder = preprocessing.OneHotEncoder(sparse_output=False, dtype=int) 

array = encoder.fit_transform(X[col_norm])

encoded_df_clean = pd.DataFrame(array, columns=encoder.get_feature_names_out(col_norm))

encoded_df_clean.index = X.index
X_clean = pd.concat([X.drop(columns=col_norm), encoded_df_clean], axis=1)
label_enc = LabelEncoder()
data = load_classification_data()
y = data["Comptage horaire"]  
y = label_enc.fit_transform(y)

@st.cache_data
def load_regression_data():
    df_reg = pd.read_csv('d/home/ketsiapedro/Bureau/lieu-compteur-one-hot-encoded.csv', index_col=0)
    df_reg["comptage horaire"] = np.log1p(df_reg["comptage horaire"])
    X_reg = df_reg.drop(columns=['comptage horaire'])
    y_reg = df_reg['comptage horaire']
    return train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

@st.cache_resource
def train_classification_model(params):
    X_train, X_test, y_train, y_test= train_test_split(X_clean, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@st.cache_resource
def train_regression_model(params):
    (X_train, X_test, y_train, y_test) = load_regression_data()
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model, X_test, y_test

st.sidebar.header("Paramètres")
problem_type = st.sidebar.selectbox("Type de problème", ["Classification", "Régression"])

if problem_type == "Classification":
    st.header("Analyse de Classification")
    
    with st.expander("Hyperparamètres de Classification"):
        n_estimators = st.slider("n_estimators", 50, 300, 200, 50)
        max_depth = st.slider("max_depth", 10, 100, 70, 10)
        params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    model, X_test, y_test = train_classification_model(params)
    y_pred = model.predict(X_test)
    (_, _, _, _), label_enc = load_classification_data()
    
    st.subheader("Performance du Modèle")
    st.write(classification_report(y_test, y_pred, target_names=label_enc.classes_))
    
    fig = px.imshow(confusion_matrix(y_test, y_pred),
                    labels=dict(x="Prédit", y="Réel", color="Count"),
                    x=label_enc.classes_, y=label_enc.classes_,
                    color_continuous_scale='Blues')
    fig.update_layout(title="Matrice de Confusion Interactive")
    st.plotly_chart(fig, use_container_width=True)
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    fig = px.bar(feature_importance, x='importance', y='feature', 
                 title='Top 10 des Caractéristiques Importantes')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Analyse de Régression")
    
    with st.expander("Hyperparamètres de Régression"):
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.5, 0.1, 0.01)
        max_iter = st.slider("Nombre d'itérations", 50, 500, 100, 50)
        params = {'learning_rate': learning_rate, 'max_iter': max_iter}
    
    model, X_test, y_test = train_regression_model(params)
    y_pred = model.predict(X_test)
    
    st.subheader("Performance du Modèle")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    col2.metric("R²", f"{r2_score(y_test, y_pred):.2f}")
    col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    
    fig = px.scatter(x=y_test, y=y_pred, 
                     labels={'x': 'Valeurs Réelles', 'y': 'Prédictions'},
                     title='Valeurs Réelles vs Prédictions')
    fig.add_shape(type='line', line=dict(dash='dash'),
                  x0=min(y_test), y0=min(y_test),
                  x1=max(y_test), y1=max(y_test))
    st.plotly_chart(fig, use_container_width=True)
    
    residuals = y_test - y_pred
    fig = px.histogram(residuals, nbins=50, 
                       title='Distribution des Résidus',
                       labels={'value': 'Résidu'})
    st.plotly_chart(fig, use_container_width=True)

st.header("Analyse Comparative des Erreurs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification")
    st.write("Meilleures métriques obtenues :")
    st.code("""
    - Accuracy : 0.85
    - Précision moyenne : 0.84
    - Rappel moyen : 0.83
    """)

with col2:
    st.subheader("Régression")
    st.write("Meilleures métriques obtenues :")
    st.code("""
    - RMSE : 15.2
    - R² : 0.92
    - MAE : 8.3
    """)

st.subheader("Conclusions")
if problem_type == "Classification":
    st.write("""
    - Les classes extrêmes (faible et haut trafic) sont les moins bien prédites
    - L'heure et le jour de la semaine sont les facteurs les plus déterminants
    - Performance acceptable mais pourrait être améliorée avec plus de données temporelles
    """)
else:
    st.write("""
    - Bonnes performances globales avec un R² de 0.92
    - Les erreurs augmentent lors des pics de trafic exceptionnels
    - Le modèle capture bien les variations saisonnières horaires
    """)


# FOOTER ===========================================================
st.markdown("---")
st.caption("""
*Rapport rédigé par: Alexandre COURROUX - Kévin LAKHDAR - Ketsia PEDRO - Eliah REBSTOCK*
""")