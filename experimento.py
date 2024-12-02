# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:15:10 2024

@author: nowok
"""
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import lightgbm as lgb
import optuna
import time 

api = KaggleApi()
api.authenticate()

# ID de la competencia
competition_id = "dm-ey-f-2024-tercera"


ruta_csv = r"C:\Users\nowok\Downloads\undersampled_202008_202109_lags1.csv"  # Reemplaza con la ruta de tu archivo

# Lee el archivo CSV
data = pd.read_csv(ruta_csv)




# Filtrar las filas correspondientes
mask = (data['foto_mes'] == 202108) & (data['clase_ternaria'] == 'BAJA+2')
# Asignar NaN solo a la columna clase_ternaria
data.loc[mask, 'clase_ternaria'] = np.nan


mes_test = 202109
X_test = data[data['foto_mes'] == mes_test]
X_test.drop(columns=['clase_ternaria'], axis=1, inplace=True)

data['clase_peso'] = 1.0

data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00003
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

data['clase_binaria1'] = 0
data['clase_binaria2'] = 0
data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)


# Filtrar los datos para entrenamiento y prueba
train_data = data[data['foto_mes'] != mes_test]


X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_train_binaria1 = train_data['clase_binaria1']
y_train_binaria2 = train_data['clase_binaria2']
w_train = train_data['clase_peso']

# Lista para guardar predicciones
y_pred_lgm = []


train_data1 = lgb.Dataset(X_train, label=y_train_binaria1, weight=w_train)
train_data2 = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)


    
def obtener_puntaje_kaggle():
    

    submissions = api.competitions_submissions_list(competition_id)
    if submissions:
        # último envío
        last_submission = max(submissions, key=lambda x: x['date'])
        return float(last_submission['publicScore']) if 'publicScore' in last_submission else None
    return None

def objective(trial):
    # Hiperparámetros a optimizar
    num_leaves = trial.suggest_int('num_leaves', 8, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.1, 1.0)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    lambda_l1 = trial.suggest_float('lambda_l1', 0.0, 10.0)
    lambda_l2 = trial.suggest_float('lambda_l2', 0.0, 10.0)
    min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 1.0)
    bagging_freq = trial.suggest_int('bagging_freq', 1, 7)

    params = {
        'objective': 'binary',
        'metric': 'custom',  
        'boosting_type': 'gbdt',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'max_depth': max_depth,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'bagging_freq': bagging_freq,
        'seed': semilla,
        'verbose': -1,
        'feature_pre_filter': False
    }


   
        
    # Entrena el modelo
    model = lgb.train(
        params,
        train_data1,  # Usa el dataset de entrenamiento
        num_boost_round=100,
        
    )
    
    # Realiza predicciones sobre X_test
# Crear una copia de X_test

# Realizar las predicciones sobre la copia de X_test
    y_pred = model.predict(X_test)

    # Selecciona los X casos más probables
    X = 12000
    X_test['Predicted'] = 0
    top_indices = np.argsort(y_pred)[::-1][:X]
    X_test.iloc[top_indices, X_test.columns.get_loc('Predicted')] = 1

    # Crear el archivo de salida
    output = X_test[['numero_de_cliente', 'Predicted']]
    file_path = f"submission_{trial.number}.csv"
    output.to_csv(file_path, index=False)

    message = f"""Hyperparameters:
                num_leaves: {num_leaves}
                learning_rate: {learning_rate}
                min_data_in_leaf: {min_data_in_leaf}
                feature_fraction: {feature_fraction}
                bagging_fraction: {bagging_fraction}
                max_depth: {max_depth}
                lambda_l1: {lambda_l1}
                lambda_l2: {lambda_l2}
                min_gain_to_split: {min_gain_to_split}
                bagging_freq: {bagging_freq}
                seed: {semilla}
                envios: {X}"""
       
    # Enviar a Kaggle
    api.competition_submit(file_path, message, competition_id)
    time.sleep(30)

    X_test.drop(columns=['Predicted'], axis=1, inplace=True)

    # Obtener la puntuación de Kaggle
    public_score = obtener_puntaje_kaggle()
    if public_score is not None:
        print(f"Trial {trial.number} - Puntuación de Kaggle: {public_score}")
    return public_score  # Esto es lo que Optuna optimizará



# Configuración del almacenamiento y optimización
storage_name = "sqlite:///experimento_kaggle.db"


# Define las semillas

semillas = [2, 17, 47, 9, 112, 15, 175, 971, 223, 22627]
semillas = [47]
semillas = [12,1,123,1234,122345]

# Iterar sobre las semillas
for semilla in semillas:
       study_name = "optuna_kaggle_study_" + str(semilla)
       study = optuna.create_study(
           direction="maximize",
           study_name=study_name,
           storage=storage_name,
           load_if_exists=True
       )


       # Optimización
       study.optimize(objective, n_trials=20)

