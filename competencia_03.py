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


ruta_csv = r"C:\Users\nowok\Downloads\undersampled_202008_202109_lags2.csv"  # Reemplaza con la ruta de tu archivo

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
    learning_rate = trial.suggest_float('learning_rate', 0.04, 0.08)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 600, 1000)
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
    X = 11750
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
    time.sleep(10)

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



# Iterar sobre las semillas
for semilla in semillas:
    study_name = "optuna_kaggle_study_" + str(semilla)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True
    )

    # Enqueue trial con los parámetros iniciales
    # Optimización
    study.optimize(objective, n_trials=40)





# Semillas a usar
semillas = [47]
# Hiperparámetros obtenidos de Optuna
params_template = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'first_metric_only': True,
    'boost_from_average': True,
    'feature_pre_filter': False,
    'max_bin': 31,
    'num_leaves': study.best_trial.params.get('num_leaves', 31),
    'learning_rate': study.best_trial.params.get('learning_rate', 0.1),
    'min_data_in_leaf': study.best_trial.params.get('min_data_in_leaf', 20),
    'feature_fraction': study.best_trial.params.get('feature_fraction', 1.0),
    'bagging_fraction': study.best_trial.params.get('bagging_fraction', 1.0),
    'max_depth': study.best_trial.params.get('max_depth', -1),
    'lambda_l1': study.best_trial.params.get('lambda_l1', 0.0),
    'lambda_l2': study.best_trial.params.get('lambda_l2', 0.0),
    'min_gain_to_split': study.best_trial.params.get('min_gain_to_split', 0.0),
    'bagging_freq': study.best_trial.params.get('bagging_freq', 0),
    'verbose': 0,
}

# Para acumular predicciones
y_pred_lgm = []

# Entrenar con distintas semillas y acumular predicciones
for semilla in semillas:
    print('procesando'+str(semilla))
    X_test = X_test[X_train.columns]

    params = params_template.copy()
    params['seed'] = semilla  # Ajustar la semilla para cada modelo
    
    # Crear dataset de entrenamiento
    train_data = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)
    
    # Entrenar el modelo
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
    )
    
    # Generar predicciones y acumular
    y_pred = model.predict(X_test)
    y_pred_lgm.append(y_pred)

# Convertir predicciones acumuladas a un array

# Realizar envíos variando X
y_pred_lgm_array = np.array(y_pred_lgm)
y_pred_avg = np.mean(y_pred_lgm_array, axis=0)

for X in range(11000, 13001, 200):  # Iterar sobre los valores de X
    X_test = X_test[X_train.columns]

    # Crear una copia del DataFrame para evitar modificar el original
    X_test_copy = X_test.copy()
    
    # Inicializar la columna 'Predicted' si no existe
    if 'Predicted' not in X_test_copy.columns:
        X_test_copy['Predicted'] = 0
    
    # Obtener los índices de las X probabilidades más altas
    top_indices = np.argsort(y_pred_avg)[::-1][:X]
    
    # Marcar los índices seleccionados como positivos
    X_test_copy.iloc[top_indices, X_test_copy.columns.get_loc('Predicted')] = 1
    
    # Crear el archivo de salida
    output = X_test_copy[['numero_de_cliente', 'Predicted']]
    file_path = f"submission_{X}.csv"
    output.to_csv(file_path, index=False)

    # Mensaje descriptivo para el envío
    message = f"""
    Ensemble con {len(semillas)} semillas.
    Mejores hiperparámetros de Optuna:
    num_leaves: {params_template['num_leaves']}
    learning_rate: {params_template['learning_rate']}
    min_data_in_leaf: {params_template['min_data_in_leaf']}
    feature_fraction: {params_template['feature_fraction']}
    bagging_fraction: {params_template['bagging_fraction']}
    max_depth: {params_template['max_depth']}
    lambda_l1: {params_template['lambda_l1']}
    lambda_l2: {params_template['lambda_l2']}
    min_gain_to_split: {params_template['min_gain_to_split']}
    bagging_freq: {params_template['bagging_freq']}
    envios: {X}
    """

    # Enviar a Kaggle
    api.competition_submit(file_path, message, competition_id)
