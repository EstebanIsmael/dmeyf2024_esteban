# -*- coding: utf-8 -*-
"""LightGBM Esteban.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VlsZbcnduM7GXScAKEw-5QF5g-Tjnskc
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install optuna==3.6.1


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna



# Ruta del archivo CSV
ruta_csv = r"C:\Users\nowok\Downloads\undersampled_lags.csv"  # Reemplaza con la ruta de tu archivo
#df = pd.read_csv(ruta_csv)


ruta_csv = r"C:\Users\nowok\Downloads\undersampled_202008_202109_lags1.csv" 
# Lee el archivo CSV
data = pd.read_csv(ruta_csv)


ganancia_acierto = 273000
costo_estimulo = 7000

data.head()


data['clase_peso'] = 1.0

data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00003
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

data['clase_binaria1'] = 0
data['clase_binaria2'] = 0
data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)

# Definir meses de entrenamiento y prueba
mes_test = 202107   

# Filtrar los datos para entrenamiento y prueba
train_data = data[~data['foto_mes'].isin([202109, 202108])]
test_data = data[data['foto_mes'] == mes_test]

X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_train_binaria1 = train_data['clase_binaria1']
y_train_binaria2 = train_data['clase_binaria2']
w_train = train_data['clase_peso']

X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_test_binaria1 = test_data['clase_binaria1']
y_test_class = test_data['clase_ternaria']
w_test = test_data['clase_peso']

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00003, ganancia_acierto, 0) - np.where(weight < 1.00003, costo_estimulo, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True

"""LGBM necesita su propio tipo de Datasets:

"""

train_data1 = lgb.Dataset(X_train, label=y_train_binaria1, weight=w_train)
train_data2 = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)

def objective(trial):
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
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
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
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        # early_stopping_rounds= int(50 + 5 / learning_rate),
        feval=lgb_gan_eval,
        stratified=True,
        nfold=5,
        seed=semilla
    )
    max_gan = max(cv_results['valid gan_eval-mean'])
    best_iter = cv_results['valid gan_eval-mean'].index(max_gan) + 1

    trial.set_user_attr("best_iter", best_iter)

    return max_gan * 5

# Define las semillas
#semillas = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]

semillas = [27, 127, 457, 79, 1112, 115, 167, 191, 2223, 252627]

#semillas = [123]
# Ruta del almacenamiento de Optuna
storage_name = "sqlite:///optuna_undersampled_lags.db"

# Iterar sobre las semillas
for semilla in semillas:
        # Nombre único del estudio basado en la semilla
        study_name = f"exp_lgbm_seed_{semilla}"

        # Crear o cargar el estudio
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )

        # Optimizar el estudio
        study.optimize(objective, n_trials=50)
        

# entreno con todo


ruta_csv = r"C:\Users\nowok\Downloads\undersampled_202008_202109_lags1.csv" 
# Lee el archivo CSV
data = pd.read_csv(ruta_csv)


ganancia_acierto = 273000
costo_estimulo = 7000

# Filtrar las filas correspondientes
mask = (data['foto_mes'] == 202108) & (data['clase_ternaria'] == 'BAJA+2')
# Asignar NaN solo a la columna clase_ternaria
data.loc[mask, 'clase_ternaria'] = np.nan



data['clase_peso'] = 1.0

data.loc[data['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00003
data.loc[data['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

data['clase_binaria1'] = 0
data['clase_binaria2'] = 0
data['clase_binaria1'] = np.where(data['clase_ternaria'] == 'BAJA+2', 1, 0)
data['clase_binaria2'] = np.where(data['clase_ternaria'] == 'CONTINUA', 0, 1)

# Definir meses de entrenamiento y prueba
# meses_train = [202012, 202011, 202010, 202009, 202008, 202007, 202101, 202102, 202103, 202104, 202105,202106]
mes_test = 202109  # Junio 2021

# Filtrar los datos para entrenamiento y prueba
train_data = data[~data['foto_mes'].isin([202109])]
test_data = data[data['foto_mes'] == mes_test]

X_train = train_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_train_binaria1 = train_data['clase_binaria1']
y_train_binaria2 = train_data['clase_binaria2']
w_train = train_data['clase_peso']

X_test = test_data.drop(['clase_ternaria', 'clase_peso', 'clase_binaria1','clase_binaria2'], axis=1)
y_test_binaria1 = test_data['clase_binaria1']
y_test_class = test_data['clase_ternaria']
w_test = test_data['clase_peso']

# Lista para guardar predicciones
y_pred_lgm = []


train_data1 = lgb.Dataset(X_train, label=y_train_binaria1, weight=w_train)
train_data2 = lgb.Dataset(X_train, label=y_train_binaria2, weight=w_train)


for semilla in semillas:
    study_name = f"exp_lgbm_seed_{semilla}"
    try:
        # Cargar el estudio correspondiente
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name
        )

        # Obtener la mejor iteración del modelo
        best_iter = study.best_trial.user_attrs.get("best_iter")
        if best_iter is None:
            print(f"Advertencia: 'best_iter' no está definido para la semilla {semilla}. Se omite.")
            continue

        print(f"Semilla {semilla} - Mejor cantidad de árboles: {best_iter}")

        # Definir los parámetros del modelo
        params = {
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
            'seed': semilla,
            'verbose': 0
        }

        # Preparar datos de entrenamiento
        train_data = lgb.Dataset(X_train,
                                 label=y_train_binaria2,
                                 weight=w_train)

        model = lgb.train(params,
                          train_data,
                          num_boost_round=best_iter)

        # Generar predicciones y acumular
        y_pred = model.predict(X_test)
        y_pred_lgm.append(y_pred)

    except Exception as e:
        print(f"Error al procesar la semilla {semilla}: {e}")


y_pred_lgm_array = np.array(y_pred_lgm)

# Calcular el promedio de las probabilidades para cada fila
# (Promedia a lo largo de los modelos, es decir, a lo largo de la primera dimensión)
y_pred_avg = np.mean(y_pred_lgm_array, axis=0)

# Número de casos a marcar como positivos
X = 12000

if 'Predicted' not in X_test.columns:
    X_test['Predicted'] = 0
    
# Obtener los índices de las X probabilidades más altas
top_indices = np.argsort(y_pred_avg)[::-1][:X]

# Marcar los índices seleccionados como positivos
X_test.iloc[top_indices, X_test.columns.get_loc('Predicted')] = 1

# Crear el archivo de salida para Kaggle
output = X_test[['numero_de_cliente', 'Predicted']]

# Guardar el archivo en formato CSV, usando un nombre que incluya la semilla o cualquier identificador relevante
nombre_archivo = 'predicciones_kaggle.csv'  # Cambiar si quieres incluir semilla u otros detalles
output.to_csv(nombre_archivo, index=False)

print(f"Archivo guardado como {nombre_archivo}")

