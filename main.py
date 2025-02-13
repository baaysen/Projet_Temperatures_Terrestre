import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


@st.cache_data
def load_temp():
    # Convertir la colonne "year" au bon format
    temp_data =pd.read_csv("data/GLB.Ts+dSST_.csv", skiprows=1)
    temp_data['Year'] = temp_data['Year'].apply(lambda x: int(str(x).replace(",", "")))
    return temp_data

# DataFrame df_co2_pays
@st.cache_data
def load_co2_pays():
    # Convertir la colonne "year" au bon format
    co2_pays_data = pd.read_csv("data/df_20_co2_pays.xls")
    co2_pays_data['year'] = co2_pays_data['year'].apply(lambda x: int(str(x).replace(",", "")))
    return co2_pays_data
 
# DataFrame df_co2_region
@st.cache_data
def load_co2_region():
    # Convertir la colonne "year" au bon format
    co2_region_data = pd.read_csv("data/df_20_co2_region_2.csv")
    co2_region_data['year'] = co2_region_data['year'].apply(lambda x: int(str(x).replace(",", "")))
    return co2_region_data

# DataFrame df_zone
@st.cache_data
def load_df_zone():
    # Convertir la colonne "year" au bon format
    df_zone = pd.read_csv("data/zone_temp.csv")
    return df_zone

# Chargement des données
df_temp = load_temp()
df_co2_pays = load_co2_pays()
df_co2_region = load_co2_region()
df_zone = load_df_zone()
# ------------------------- CODES -----------------------------------------------------------------
# 
# Fusion des datasets
merged_data = pd.merge(df_co2_pays, df_temp, left_on='year', right_on='Year', how='inner')
# Convertir 'J-D' en float
merged_data['J-D'] = pd.to_numeric(merged_data['J-D'])
# Evolution des températures au cours des années
merged_data_temp = merged_data.groupby('year')['J-D'].mean()
# ------------------------------------------------------------------------------------------------

# Ajout d'un titre + création des pages streamlit
st.title("Température Terrestre")
st.sidebar.title("Sommaire")
pages=["Introduction", 
       "Exploration & DataViz Temperatures", 
       "Exploration & DataViz CO2", 
       "Modélisations & Prédiction", 
       "Conclusion & Perspective"]
page=st.sidebar.radio("Aller vers", pages)
# Ajout des auteurs du projet
st.sidebar.title("Auteurs")
# st.sidebar.write("A. Diakhaté") # Sans le lien vers LinkedIn
st.sidebar.markdown("[A. Diakhaté](https://www.linkedin.com/in/a-diakhate-5a998265/)")
st.sidebar.write("A. De-Polignac")
st.sidebar.write("H. Dridi")

# ---------------------------- INTRODUCTION
if page == pages[0] : 
  st.write("### Introduction")
  st.write("- Augementation des températures moyennes depuis plus d'un siècle")
  st.write("- Causé en grande partie par les activités humaine")
  st.write("- Modification des écosystèmes et des équilibres météorologie")
  st.image("data/intro.gif", caption="Illustration de l'introduction")

  st.write("#### Objectifs:")
  st.write("- Analyse des données de variations de températures collectées par la NASA")
  st.write("- Mise en évidence d'un lien de causalité (s’il existe) entre l’augmentation des émissions de CO2 et la variation des températures")
  st.write("- Modélisation et Prédiction du réchauffement climatique pour horizon 2050")

# ---------------------------- EXPLORATION & DATAVIZ TEMPERATURE
# Affichage des df temperature
if page == pages[1] :                         # Page température 

  st.write("### Dataset température")
  st.write("Source: https://data.giss.nasa.gov/gistemp/")
    # Création des onglets pour température 
  tab1, tab2, tab3 = st.tabs(["Dataset temp", "Traitement", "Visualisations"])

  with tab1:                                  # onglet Dataset température / taille  / stats 
    st.subheader("Dataset temp")
    st.dataframe(df_temp)
    st.write(df_temp.shape)
    st.dataframe(df_temp.describe())
  with tab2:                                 # onglet Traitement
    st.subheader("Traitement")
    st.write("- Dataset 1: Variations annuelles et mensuelles des température globales")
    st.write("Problèmatique : Les données initiales contenaient des valeurs nulles qui devaient être traitées pour assurer une analyse précise")
    st.image("data/var_nans_dataset_temp.JPG", caption="Variables et taux de NaNs")
    st.write("- Dataset 2: Variations annuelles des températures globales et par hémisphère (Nord/Sud)")
    st.image("data/var_dataset2_temp.JPG", caption="Illustration de l'introduction")
    st.write("Traitement: Interpolation linéaire")
    st.write("- Conservation des tandences et de la variabilité des résultats")
    st.write("- Conservation de la continuité des données sans introduire de biais significatifs")

  with tab3:                                 # onglet visualisation
    st.subheader("Visualisations")
    st.write("Ce graphique nous montre bien une augmentation progressive des températures depuis les années 1880 puisque cette variation est\
               négative avant la période de référence 1950-80 puis devient positive après cette dernière. On observe aussi une accélération \
              de la variation avec une pente de courbe qui augmente dans le temps")
    
# Visualisation plotly
    fig = px.line(merged_data_temp, x=merged_data['year'], y=merged_data['J-D'], markers=True,
                  labels={'Année', 'Température globale annuelle (°C)'},
                  title='Évolution des températures globales annuelles (1880-2022)')
    fig.update_layout(xaxis_title='Année', yaxis_title='Température (°C)')
    st.plotly_chart(fig)
    st.write("Benchmarking: Giec, 1er groupe de travail, 2021 et HadCrut 5")
    st.image("data/Benchmark_giec.jpg", caption="")

#------------------------------------- DataViz température par tranche de 40 ans
#
# Convertir l'année en datetime et définir comme index
    df_temp['Date'] = pd.to_datetime(df_temp['Year'], format='%Y')
    df_temp.set_index('Date', inplace=True)

# Nettoyer les données : remplacer les valeurs non numériques par NaN et convertir en numérique
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        df_temp[month] = pd.to_numeric(df_temp[month], errors='coerce')

# Définir les périodes
    periods = [1880, 1920, 1960, 2000, 2020]

# Liste pour stocker les moyennes mensuelles par période
    monthly_means_periods = []

# Calculer les moyennes mensuelles pour chaque période
    for start, end in zip(periods[:-1], periods[1:]):
        start_date = pd.Timestamp(start, 1, 1)
        end_date = pd.Timestamp(end, 1, 1)
        period_df = df_temp.loc[start_date:end_date]
        monthly_means_period = period_df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']].mean()
        monthly_means_periods.append(monthly_means_period)

# Créer un DataFrame avec les résultats
    monthly_means_df = pd.DataFrame(monthly_means_periods, index=[f'{start}-{end-1}' for start, end in zip(periods[:-1], periods[1:])])

# Tracer les moyennes mensuelles par période sur un même graphique avec Plotly
    fig = go.Figure()

    months = ['Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec']
    x = list(range(1, 13))  # Mois de 1 à 12

    for i, means in enumerate(monthly_means_periods):
        label = f"{periods[i]}-{periods[i+1]-1}"
        fig.add_trace(go.Scatter(x=x, y=means, mode='lines+markers', name=label))

# Ajouter les labels et le titre
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x,
            ticktext=months
        ),
        yaxis_title='Variation des températures (°C)',
        xaxis_title='Mois',
        title='Températures moyennes mensuelles par période de 40 ans',
        showlegend=True
    )

# Ajouter une ligne horizontale à 0
    fig.add_shape(type="line", x0=1, y0=0, x1=12, y1=0,
                  line=dict(color="Black",width=1, dash="dash"))

# Afficher le graphique avec Streamlit
    st.plotly_chart(fig)
# Commentaire
    st.write("- A partir de 1980:  Les écarts entre les températures semblent se creuser progressivement.")
    st.write("- Cette observation suggère une accélération des variations de température au fil du temps.")

#--------------------------------- DataViz température zone

# Convertir 'Year' en type datetime
    df_zone['Year'] = pd.to_datetime(df_zone['Year'], format='%Y').dt.year

# Suppression des colonnes par latitude (4 à 14)
    df_zone = df_zone.drop(df_zone.columns[4:15], axis=1)

# Observation des années 1980 à aujourd'hui
    df_zone_1980 = df_zone[df_zone['Year'] >= 1980].copy()

# Calcul de la moyenne de 'Glob' depuis 1980
    mean_value = df_zone_1980['Glob'].mean()

# Tracer les données avec Plotly
    fig = go.Figure()

# Tracer 'Glob', 'NHem', 'SHem'
    fig.add_trace(go.Scatter(x=df_zone_1980['Year'], y=df_zone_1980['Glob'], mode='lines', name='Glob'))
    fig.add_trace(go.Scatter(x=df_zone_1980['Year'], y=df_zone_1980['NHem'], mode='lines', name='NHem'))
    fig.add_trace(go.Scatter(x=df_zone_1980['Year'], y=df_zone_1980['SHem'], mode='lines', name='SHem'))

# Ajouter une ligne horizontale à 0
    fig.add_shape(type="line", x0=df_zone_1980['Year'].min(), y0=0, x1=df_zone_1980['Year'].max(), y1=0,
                  line=dict(color="Black", width=1, dash="dash"))

# Ajouter la ligne de moyenne
    fig.add_shape(type="line", x0=df_zone_1980['Year'].min(), y0=mean_value, x1=df_zone_1980['Year'].max(), y1=mean_value,
                  line=dict(color="Red", width=1, dash="dot"), name='Moyenne')

# Mettre à jour les labels et le titre
    fig.update_layout(
        xaxis_title='Année',
        yaxis_title='Variations des températures (°C)',
        title='Variations des températures vs. la période de référence',
        showlegend=True
    )

# Ajouter la légende pour la ligne de moyenne
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='Red', width=1, dash='dot'),
        name='Moyenne'
    ))

# Afficher le graphique avec Streamlit
    st.plotly_chart(fig)

 

# ---------------------------- EXPLORATION & DATAVIZ CO2

if page == pages[2] :                         # Page CO2 
  
  st.write("Source: https://github.com/owid/co2-data")
  st.write("### Processing & DataViz: CO2")
  # Création des onglets pour CO2
  tab10, tab20, tab30, tab40 = st.tabs(["Dataset pays", "Dataset region", "Traitement", "Visualisations"])

  with tab10: # onglet Dataset pays
    st.subheader("Dataset pays")
    st.write(df_co2_pays)
    st.write(df_co2_pays.shape)
    st.dataframe(df_co2_pays.describe())
  with tab20:# onglet Dataset region
    st.subheader("Dataset region")
    st.write(df_co2_region)
    st.write(df_co2_region.shape)
    st.dataframe(df_co2_region.describe())
  with tab30: # onglet Traitement CO2
     st.subheader("Traitement")
     st.write("- Outliers: On observe une masse de données en dessous de 5k et plusieurs valeurs abérrantes")
     st.image("data/boxplot_co2.JPG", caption="Boxplot des emissions CO2 par année")
     st.write("- Statistiques: Moyenne = 869k / Médiane = 12k => Nous avons des valeurs qui viennent fausser l’analyse")
     st.image("data/stat_co2.JPG", caption="Statistiques Dataset CO2 'brute'")
     st.write("- Variables: Présence de données régionales ==> séparation en deux datasets: Region/Country")
     st.image("data/variables_co2.JPG", caption="Variablse Dataset CO2")
     st.write("- Traitement: NaNS = 0")
  with tab40: # onglet Visualisations
    st.subheader("Visualisations")
    st.write("Cibler les 20 variables les plus corrélées avec la variable CO2")
    st.write("Passer d'un DataFrame de 78 à 21 colonnes")
    st.write("Cibler les 20 variables les plus corrélées avec la variable CO2")
  # Sélection des variables quantitatives uniquement
    df_var_quantitative = merged_data.select_dtypes(include=['int', 'float'])

    # Suppression des variables 'year' et 'population'
    labels = ['year', 'population'] 
    df_var_quantitative = df_var_quantitative.drop(labels, axis=1)
    
    # Calcul de la matrice de corrélation avec 'co2'
    correlation_co2_region = df_var_quantitative.corr()['co2']

    # Sélection des 20 variables les plus corrélées avec 'co2'
    top_20_corr_vars_region = correlation_co2_region.abs().nlargest(21)

    # Sélection des noms des colonnes des 20 variables les plus corrélées avec 'co2'
    top_20_corr_vars_names = top_20_corr_vars_region.index

    # Extraction des données correspondant aux 20 variables les plus corrélées avec 'co2'
    df_top_20_corr_vars = df_var_quantitative[top_20_corr_vars_names]

    # Calcul de la matrice de corrélation entre ces 20 variables
    matrice_corr_top_20_vars = df_top_20_corr_vars.corr()
 
    # Tracé du heatmap des 20 variables les plus corrélées avec 'co2' avec Plotly
    fig_heatmap = px.imshow(matrice_corr_top_20_vars, 
                            labels=dict(x="Variables", y="Variables", color="Corrélation"),
                            x=matrice_corr_top_20_vars.columns,
                            y=matrice_corr_top_20_vars.columns,
                            title='Heatmap des 20 variables les plus corrélées avec "co2"',
                            color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heatmap)

    st.write("Le CO2 est majoritaire sur l'augementation des températures")
    st.image("data/influence_des_gaz.JPG", caption="Influence des Gaz sur le réchauffement climatique")
    st.write("==> Modélisations: CO2 / Température")
 
#-----------------------MODELISATION CO2 & TEMPERATURE

# ---------------------------- MODÉLISATION CO2
if page == pages[3] :                         # Page modélisation CO2

  tab100, tab200, tab300 = st.tabs(["Modélisation CO2", "Modélisation Température", "Prédiction"])

  with tab100:
    st.subheader("Modélisation CO2")
    
    # Sélectionner la variable cible et les variables explicatives
    variable_cible = 'co2'  # principal gaz à effet de serre
    variables_explicatives = ['population', 'gdp', 'total_ghg', 'primary_energy_consumption',
                              'temperature_change_from_co2', 'temperature_change_from_ch4', 'temperature_change_from_n2o']

    # Séparer les variables explicatives (X) et la variable cible (y)
    X = df_co2_pays[variables_explicatives]
    y = df_co2_pays[variable_cible]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser le modèle
    model_rf = RandomForestRegressor(random_state=42)

    # Entraîner le modèle sur les données d'entraînement
    model_rf.fit(X_train, y_train)

    # Prédire les valeurs sur l'ensemble de test
    y_pred_rf = model_rf.predict(X_test)

    # Évaluer les performances du modèle
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)  # On prend la racine carrée manuellement

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    # Calculer l'importance des variables
    feature_importances = model_rf.feature_importances_

    # Créer un DataFrame pour une meilleure visualisation
    feat_importances = pd.DataFrame(feature_importances, index=X.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)

   

    # Visualiser l'importance des variables avec Plotly
    fig = px.bar(feat_importances, x=feat_importances.index, y='Importance', title='Importance des variables', labels={'index': 'Variables', 'Importance': 'Importance'})
    st.plotly_chart(fig)

#-----------COMPARAISON R²

    # Données des résultats
    resultats = pd.DataFrame({
        'Algorithme': ['Régression Linéaire', 'Random Forest', 'Decision Tree',
                      'Régression Linéaire', 'Random Forest', 'Decision Tree'],
        'Essai': [1, 1, 1, 2, 2, 2],
        'R^2': [0.934, 0.997, 0.996, 0.889, 0.996, 0.988]
    })

    # Créer le graphique avec Plotly
    fig = px.bar(resultats, x='Algorithme', y='R^2', color='Essai', barmode='group',
                color_discrete_map={1: 'orange', 2: 'blue'},
                title='Performances des algorithmes (R^2)',
                labels={'R^2': 'Score R²', 'Algorithme': 'Algorithme'})

    fig.update_layout(yaxis=dict(range=[0, 1]), xaxis=dict(tickangle=-45), legend_title_text='Essai')

    # Afficher le graphique avec Streamlit
    st.title('Comparaison des Performances des Algorithmes')
    st.plotly_chart(fig)

#-----------COMPARAISON MSE

    # Données des résultats
    resultats = pd.DataFrame({
        'Algorithme': ['Régression Linéaire', 'Random Forest', 'Decision Tree',
                      'Régression Linéaire', 'Random Forest', 'Decision Tree'],
        'Essai': [1, 1, 1, 2, 2, 2],
        'MSE': [8290.4, 326.9, 460.23, 14084.8, 523.7, 1529.9]
    })

    # Créer le graphique avec Plotly
    fig = px.bar(resultats, x='Algorithme', y='MSE', color='Essai', barmode='group',
                color_discrete_map={1: 'orange', 2: 'blue'},
                title='Performances des algorithmes (MSE)',
                labels={'MSE': 'Erreur Quadratique Moyenne (MSE)', 'Algorithme': 'Algorithme'})

    fig.update_layout(xaxis=dict(tickangle=-45), legend_title_text='Essai')

    # Afficher le graphique avec Streamlit
    st.title('Comparaison des Performances des Algorithmes (MSE)')
    st.plotly_chart(fig)


#----------------CHOIX MODELE CO2
    # Afficher les résultats avec Streamlit
    st.write("Modélisation et analyse des émissions de CO2")
    st.write("### Évaluation du modèle Random Forest")
    st.write(f"Mean Squared Error (MSE): {mse_rf:.3f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_rf:.3f}")
    st.write(f"R² Score: {r2_rf:.3f}")
    st.write(f"Mean Absolute Error (MAE): {mae_rf:.3f}")
    # Tracer les valeurs réelles et les prédictions avec Plotly
    scatter_fig = px.scatter(x=y_test, y=y_pred_rf, labels={'x': 'Valeurs réelles', 'y': 'Prédictions'}, title="Prédictions du modèle Random Forest Regressor")
    scatter_fig.add_shape(type="line", line=dict(dash='dash'), x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
    st.plotly_chart(scatter_fig)

 #--------------------------------- MODELISATION TEMPERATURES 
  with tab200:
    st.subheader("Modélisation Température")
    st.write("### Algorithmes Évalués:")
    st.write("- Régression Linéaire")
    st.write("- Random Forest")
    st.write("- Decision Tree")
    st.write("- Gradient Boosting")
    
#-------------------------Comparaison des Algo pour modélisation températures

# Données des résultats
    resultats = pd.DataFrame({
        'Algorithme': ['Régression Linéaire', 'Random Forest', 'Decision Tree', 'Gradient Boosting',
                      'Régression Linéaire', 'Random Forest', 'Decision Tree', 'Gradient Boosting'],
        'Essai': [1, 1, 1, 1, 2, 2, 2, 2],
        'R^2': [0.053, 0.795, 0.705, 0.727, 0.062, 0.795, 0.705, 0.727]
    })

# Créer le graphique avec Plotly
    fig = px.bar(resultats, x='Algorithme', y='R^2', color='Essai',
                color_discrete_map={1: 'orange', 2: 'blue'},
                barmode='group',
                title='Performances des algorithmes (R^2)',
                labels={'Essai': 'Essai', 'Algorithme': 'Algorithme', 'R^2': 'R^2'},
                category_orders={"Algorithme": ['Régression Linéaire', 'Random Forest', 'Decision Tree', 'Gradient Boosting']}
                )

    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        legend_title_text='Essai'
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig)

#----------------CHOIX MODELISATION TEMPERATURE :  RandomForest

# Sélectionner la variable cible et les variables explicatives
    variable_cible = 'J-D'  # Température globale annuelle
    variables_explicatives = ['population', 'co2', 'total_ghg', 'primary_energy_consumption',
                              'temperature_change_from_co2', 'temperature_change_from_ch4',
                              'temperature_change_from_n2o', 'gdp', 'co2_including_luc',
                              'coal_co2', 'oil_co2', 'gas_co2']

    X = merged_data[variables_explicatives]
    y = merged_data[variable_cible]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
    model_rf = RandomForestRegressor(random_state=42)
    model_rf.fit(X_train, y_train)

# Faire des prédictions
    y_pred_rf = model_rf.predict(X_test)

# Évaluer les performances
    mse_rf_ = mean_squared_error(y_test, y_pred_rf)
    rmse_rf_ = np.sqrt(mse_rf_ )  # On prend la racine carrée manuellement
    r2_rf_ = r2_score(y_test, y_pred_rf)

# Tracer les valeurs réelles et les prédictions avec Plotly
    fig = px.scatter(x=y_test, y=y_pred_rf, labels={'x': 'Valeurs réelles', 'y': 'Prédictions'},
                    title="Prédictions du modèle Random Forest Regressor")
    fig.update_layout(showlegend=False)

# Affichage avec Streamlit
    st.title("Modélisation et prédictions")
    st.write("### Évaluation du modèle Random Forest Regressor")
    st.write(f"Mean Squared Error (MSE): {mse_rf_:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_rf_:.4f}")
    st.write(f"R² Score: {r2_rf_:.4f}")

    st.plotly_chart(fig)


# ---------------------------------- PREDICTION  
  with tab300:
    st.subheader("Prédiction")
    st.write("Ajout de la variable [year] => Autres modélisations")
    st.write("Choix: LinearRegression")


    # Dictionnaire des scores R²
    dict_algo = {
        'RandomForestRegressor': 1.000,
        'Régression linéaire': 0.771,
        'DecisionTreeRegressor': 1.000,
        'Gradient Boosting': 0.985
    }

    # Création du DataFrame à partir du dictionnaire
    df_algo = pd.DataFrame.from_dict(dict_algo, orient='index', columns=['R^2'])

    # Créer un graphique en barres des scores R² avec Plotly
    fig_r2 = px.bar(df_algo, x=df_algo.index, y='R^2', 
                    labels={'index': 'Modèles Machine Learning', 'R^2': 'Score R^2'},
                    title='Scores R² en fonction des modélisations testées',
                    text='R^2')
    
    # Personnaliser les couleurs et les labels
    fig_r2.update_traces(marker_color=['blue', 'green', 'red', 'purple'], texttemplate='%{text:.3f}', textposition='outside')
    fig_r2.update_layout(xaxis_title='Modèles Machine Learning', yaxis_title='Score R^2', yaxis=dict(range=[0, 1.1]), 
                          uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig_r2)
    st.write("La Régression Linéaire semble etre mieux adapté pour faire la prédiction=> R² = 0.771 (moins de risque de surapprentissage)")

    # ---------------------------------- CODE PREDICTION

    # Sélectionner la variable cible et les variables explicatives
    variable_cible = 'J-D'  # Température globale annuelle
    variables_explicatives = ['year', 'population', 'co2', 'total_ghg', 'primary_energy_consumption',
                              'temperature_change_from_co2', 'temperature_change_from_ch4',
                              'temperature_change_from_n2o', 'gdp', 'co2_including_luc',
                              'coal_co2', 'oil_co2', 'gas_co2']

    X = merged_data[variables_explicatives]
    y = merged_data[variable_cible]

    # Séparation des données en un jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model_lin = LinearRegression()
    model_lin.fit(X_train, y_train)

    # Faire des prédictions
    y_pred_lin = model_lin.predict(X_test)
    # Générer les années de 2023 à 2050
    years = np.arange(2023, 2051)

    # Hypothèses des tendances pour chaque variable explicative sensible à l'année
    population_trend = 8e9 + (years - 2023) * 0.05e9  # Population augmente de 50 millions par an
    co2_trend = 36e9 + (years - 2023) * 0.30e9        # CO2 augmente de 0.3 milliards de tonnes par an
    total_ghg_trend = 50e9 + (years - 2023) * 0.3e9   # GES totaux augmentent de 0.3 milliards de tonnes par an
    energy_trend = 600e9 + (years - 2023) * 5e9       # Consommation énergétique primaire augmente de 5 milliards de tonnes par an
    temp_change_co2_trend = 0.02 + (years - 2023) * 0.001  # Température augmente de 0.001°C par an à cause du CO2
    temp_change_ch4_trend = 0.01 + (years - 2023) * 0.0005  # Température augmente de 0.0005°C par an à cause du CH4
    temp_change_n2o_trend = 0.005 + (years - 2023) * 0.0002  # Température augmente de 0.0002°C par an à cause du N2O
    gdp_trend = 100e12 + (years - 2023) * 2e12        # PIB augmente de 2 trillions par an
    co2_luc_trend = 42e9 + (years - 2023) * 0.2e9     # CO2 incluant LUC augmente de 0.2 milliards de tonnes par an
    coal_co2_trend = 20e9 + (years - 2023) * 0.1e9    # CO2 du charbon augmente de 0.1 milliards de tonnes par an
    oil_co2_trend = 15e9 + (years - 2023) * 0.05e9    # CO2 du pétrole augmente de 0.05 milliards de tonnes par an
    gas_co2_trend = 5e9 + (years - 2023) * 0.02e9     # CO2 du gaz augmente de 0.02 milliards de tonnes par an

    # Création du DataFrame avec les projections
    future_data = pd.DataFrame({
        'year': years,
        'population': population_trend,
        'co2': co2_trend,
        'total_ghg': total_ghg_trend,
        'primary_energy_consumption': energy_trend,
        'temperature_change_from_co2': temp_change_co2_trend,
        'temperature_change_from_ch4': temp_change_ch4_trend,
        'temperature_change_from_n2o': temp_change_n2o_trend,
        'gdp': gdp_trend,
        'co2_including_luc': co2_luc_trend,
        'coal_co2': coal_co2_trend,
        'oil_co2': oil_co2_trend,
        'gas_co2': gas_co2_trend,
    })

    # Prédictions pour les années futures jusqu'en 2050
    future_predictions = model_lin.predict(future_data)

    # Combinaison de l'année en fonction de la prédiction
    predictions_df = future_data[['year']].copy()
    predictions_df['J-D Prediction'] = future_predictions/1e6 # correction des echelles de température

    # Création du graphique avec Plotly
    fig = go.Figure()

    # Ajout des données historiques
    fig.add_trace(go.Scatter(x=merged_data['year'], y=merged_data['J-D'], mode='lines', name='Données globales (1880 - 2022)', line=dict(color='blue')))

    # Ajout des prédictions futures
    fig.add_trace(go.Scatter(x=predictions_df['year'], y=predictions_df['J-D Prediction'], mode='lines', name='Prédictions (2023-2050)', line=dict(color='red', dash='dash')))

    # Mise en forme du graphique
    fig.update_layout(
        title='Prédictions des températures globales annuelles (2023-2050)',
        xaxis_title='Year',
        yaxis_title='Temperature (°C)',
        legend_title='Légende',
        template='plotly_white'
    )
    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)
    st.write("Benchmarking: Horizon 2050")
    st.image("data/prediction_2050.JPG", caption="Source : Giec, 6e rapport, 2022")


# ----------------- CONCLUSION & PERSPECTIVES

if page == pages[4] :   
   st.write("### Objectifs remplis:")
   st.write("- Mise en évidence de l’accélération de l’augmentation des température dans les dataviz")
   st.write("- Forte corrélation entre émissions de CO2 et variation des températures")
   st.write("- Réalisation d’une prédiction de la hausse grâce au dernier modèle (que pour l’oral)")
   st.write("- Comparaison avec des résultats scientifiques satisfaisantes")
   st.write("- Amélioration du modèle possible grâce à l’ajout de features et affinement des hypothèses")

   st.write("### Difficultés rencontrées:")
   st.write("          SCIENTIFIQUE")
   st.write("- Facteurs d’émissions")
   st.write("- Accès à des bases de données qui les regroupent")
   st.write("          PROJET")
   st.write("- Dataset sur le CO2 difficile à traiter")
   st.write("- Modélisation: tentative de prédictions mais qui pourrait être améliorée")
   st.write("- Application immédiate de nouvelles notions")
