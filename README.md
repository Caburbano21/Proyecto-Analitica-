# Proyecto-Analitica-
código del proyecto

PROYECTO 1
CAMILO BURBANO 206754

import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns



 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
 
from pylab import rcParams
 
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
 
from collections import Counter


%matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import TomekLinks
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification

import statsmodels.api as sm

%matplotlib inline
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
# import statsmodels.formula.api as sm
import matplotlib.pylab as plt

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

# from statsmodels.formula.api import logit

import statsmodels
print(statsmodels.__version__)

from statsmodels.api import Logit





# Importando la base de Datos
path = r"C:/Users/HP/Downloads/"
# df= pd.read_csv(path + "PublicHospitalDataset.csv", encoding='utf-8', sep= ',' ) # LLamar a la base
df= pd.read_excel(path + "PublicHospitalDataset.xlsx") # LLamar a la base
df
#encoding='cp1252'
# Número de observaciones y de variables realizadas
print(df.shape)

# Tipo de variable
df.dtypes
# Ver las categorías de cada variable
print (df.Stroke.value_counts().sort_index())
print (df.Married.value_counts().sort_index())
print (df.Work.value_counts().sort_index())
print (df["Heart Disease"].value_counts().sort_index())
print (df.Residence.value_counts().sort_index())




# Ver las categorías de cada variable

print (df.Smoking.value_counts().sort_index())
print (df.Gender.value_counts().sort_index())
print (df.Hypertension.value_counts().sort_index())

df.describe()
# Estadística descriptiva de la variable 
pd.DataFrame((df.Avg_glucose_level.describe()))


# Calcular los cuartiles de la columna "Avg_glucose_level"
q1 = np.percentile(df.Avg_glucose_level, 25)
q2 = np.percentile(df.Avg_glucose_level, 50)
q3 = np.percentile(df.Avg_glucose_level, 75)


# # Crear el gráfico y ajustar el tamaño
plt.figure(figsize=(10, 8))
plt.boxplot(df.Avg_glucose_level, vert=True)

# Agregar etiquetas con los valores de los cuartiles
plt.text(0.85, q1, "Q1: {:.2f}".format(q1), bbox=dict(facecolor='red', alpha=0.5))
plt.text(0.85, q2, "Q2: {:.2f}".format(q2), bbox=dict(facecolor='green', alpha=0.5))
plt.text(0.85, q3, "Q3: {:.2f}".format(q3), bbox=dict(facecolor='blue', alpha=0.5))

# Añadir un título al gráfico
plt.title(" Caja de Bigotes de Avg Glucose Level")

# Mostrar el gráfico
plt.show()

# Gráfico de barras de la variable en cuestión
plt.hist(df.Avg_glucose_level,bins=20)

# se forma una campana en ambos lados por ende si son representativos y significativos

# Añadir un título al gráfico
plt.title(" Gráfico de Frecuencias de la variable Avg Glucose")

plt.figure(figsize=(10, 8))

# Tabla de frecuencias de la variable Average
bins = [55.12, 82.2, 109.28, 136.36, 163.64, 190.52, 217.06, 244.68, 271.74]



# Generar una tabla de frecuencia relativa con los intervalos definidos
freq_table_rel = pd.cut(df['Avg_glucose_level'], bins=bins, include_lowest=True).value_counts(normalize=True, sort=False)

# Mostrar la tabla de frecuencia relativa
pd.DataFrame(freq_table_rel)
sns.distplot(df.Avg_glucose_level)
df_2= df
df_2
# Tabla de frecuencia de Work
tabla_frec = df_2.Work.value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec["Work"] /sum(tabla_frec["Work"])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec

# Vamos a borrar los datos atipicos de la columna de Work

df_2= df_2[df_2["Work"]!="sdfsdf"]
df_2= df_2[df_2["Work"]!="dsfsdf"]
df_2= df_2[df_2["Work"]!="k–sdh-k"]
df_2= df_2[df_2["Work"]!="sdsd-i"]
df_2
print(df_2.shape)
df_2.Work.value_counts()
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Work'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(5))

# Añadir un título al gráfico
plt.title("Frecuencia variable Work")

# Añadir una etiqueta al eje x
plt.xlabel("Work")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()

# Estadistica descriptiva de Bmi
pd.DataFrame(df_2.Bmi.describe())#.to_excel(path + "Tabla.xlsx")
# Los Nan representan un 4% de la informacion , vamos a reemplazarlos con la media de la columna
df_2= df_2[df_2["Bmi"]!= 40000]
# df_2['Bmi'] = df_2['Bmi'].fillna(df_2['Bmi'].mean())
# df_2.head(50)#.dtypes
# df_2.Bmi.value_counts()
# Añadir un título al gráfico
plt.figure(figsize=(10, 8))
plt.hist(df_2.Bmi,bins=100)
plt.title(" Gráfico de Frecuencias de la variable Bmi")
plt.show()


bins = [10.30, 21.21, 32.12, 43.03, 53.95, 64.86, 75.77, 86.68, 97.6]


# Generar una tabla de frecuencia relativa con los intervalos definidos
freq_table_rel = pd.cut(df['Bmi'], bins=bins, include_lowest=True).value_counts(normalize=True, sort=False)
# Mostrar la tabla de frecuencia relativa
pd.DataFrame(freq_table_rel)

# Los Nan representan un 4% de la informacion , vamos a reemplazarlos con la media de la columna y se eliminan datos atipicos que representan menos del 1%
df_2['Bmi'] = df_2['Bmi'].fillna(df_2['Bmi'].mean())
df_2= df_2[df_2["Bmi"]<53.95]

# Añadir un título al gráfico
plt.figure(figsize=(10, 8))
plt.hist(df_2.Bmi)
plt.title(" Gráfico de Frecuencias de la variable Bmi")
plt.show()
pd.DataFrame(df_2.Bmi.describe())

# Caja de Bigotes de Bmi

# Calcular los cuartiles de la columna "Bmi"
q1 = np.percentile(df_2.Bmi, 25)
q2 = np.percentile(df_2.Bmi, 50)
q3 = np.percentile(df_2.Bmi, 75)


# # Crear el gráfico y ajustar el tamaño
plt.figure(figsize=(10, 8))
plt.boxplot(df_2.Bmi, vert=True)

# Agregar etiquetas con los valores de los cuartiles
plt.text(0.85, q1, "Q1: {:.2f}".format(q1), bbox=dict(facecolor='red', alpha=0.5))
plt.text(0.85, q2, "Q2: {:.2f}".format(q2), bbox=dict(facecolor='green', alpha=0.5))
plt.text(0.85, q3, "Q3: {:.2f}".format(q3), bbox=dict(facecolor='blue', alpha=0.5))

# Añadir un título al gráfico
plt.title(" Caja de Bigotes de Bmi")

# Mostrar el gráfico
plt.show()
pd.DataFrame(df.Age.describe())

# Calcular los cuartiles de la columna "Age"
q1 = np.percentile(df_2.Age, 25)
q2 = np.percentile(df_2.Age, 50)
q3 = np.percentile(df_2.Age, 75)


# # Crear el gráfico y ajustar el tamaño
plt.figure(figsize=(10,8))
plt.boxplot(df_2.Age, vert=True)

# Agregar etiquetas con los valores de los cuartiles
plt.text(0.85, q1, "Q1: {:.2f}".format(q1), bbox=dict(facecolor='red', alpha=0.5))
plt.text(0.85, q2, "Q2: {:.2f}".format(q2), bbox=dict(facecolor='green', alpha=0.5))
plt.text(0.85, q3, "Q3: {:.2f}".format(q3), bbox=dict(facecolor='blue', alpha=0.5))

# Añadir un título al gráfico
plt.title(" Caja de Bigotes de Age")

# Mostrar el gráfico
plt.show()
# Grafico de frecuencias de Age
tabla_frec = df_2.Age.value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec["Age"] /sum(tabla_frec["Age"])
#tabla_frec.to_excel(path + "tabla de frecuencia age.xlsx")
tabla_frec
plt.figure(figsize=(10,8))
# Añadir un título al gráfico
plt.title(" Grafico de Frecuencias Age")
plt.hist(df_2.Age,bins=20)
plt.show()
# Tabla de Frecuencias de Gender y eliminación de dato atípico Other

df_2= df_2[df_2["Gender"]!="Other"]
df_2
print(df_2.shape)
df_2.Gender.value_counts()
# Tabla de Frecuencia relativa ya depurada
tabla_frec = df_2.Gender.value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec["Gender"] /sum(tabla_frec["Gender"])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Gender'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Gender")

# Añadir una etiqueta al eje x
plt.xlabel("Gender")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()

# Tabla de Frecuencia relativa de la variable Heart Disease
tabla_frec = df_2['Heart Disease'].value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec['Heart Disease'] /sum(tabla_frec['Heart Disease'])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Heart Disease'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Heart Disease")

# Añadir una etiqueta al eje x
plt.xlabel("Heart Disease")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()
# Tabla de frecuencia relativa de la variable Hypertension
tabla_frec = df_2['Hypertension'].value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec['Hypertension'] /sum(tabla_frec['Hypertension'])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Hypertension'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Hypertension")

# Añadir una etiqueta al eje x
plt.xlabel("Hypertension")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()
# Tabla de Frecuencia de la variable Married
tabla_frec = df_2['Married'].value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec['Married'] /sum(tabla_frec['Married'])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Married'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Married")

# Añadir una etiqueta al eje x
plt.xlabel("Married")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()
# Tabla de frecuencia de la variable Residence
tabla_frec = df_2['Residence'].value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec['Residence'] /sum(tabla_frec['Residence'])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Residence'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Residence")

# Añadir una etiqueta al eje x
plt.xlabel("Residence")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()
# Tabla de Frecuencia de la variable Smoking
tabla_frec = df_2['Smoking'].value_counts().to_frame()
tabla_frec["Frec"] =tabla_frec['Smoking'] /sum(tabla_frec['Smoking'])
tabla_frec.to_excel(path + "Tabla.xlsx")
tabla_frec
# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Smoking'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# # Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# # Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# # Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(4))

# Añadir un título al gráfico
plt.title("Frecuencia variable Smoking")

# Añadir una etiqueta al eje x
plt.xlabel("Smoking")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()
# Crear un gráfico de distribución con función de densidad para la variable cuantitativa
sns.displot(df_2.Avg_glucose_level, kde=True)
# Crear un gráfico de distribución con función de densidad para la variable cuantitativa
sns.displot(df_2.Bmi, kde=True)
# Crear un gráfico de distribución con función de densidad para la variable cuantitativa
sns.displot(df_2.Age, kde=True)
matriz_correlacion = df_2.corr()
matriz_correlacion
# Crear una figura con un tamaño de 15 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm')

# Añadir un título al gráfico
plt.title("Grafico de Correlaciones")
plt.show()
# Base de Datos ya depurada
df_2 = df_2.reset_index()
df_2=df_2.drop(columns=["index"])
df_2
# Base de Datos asigna valores de cada variable en base a sus distintas categorías
# agregar etiquetas a la variable categórica
df_2['Married'].replace({'Yes': 1, 'No':0}, inplace=True)
df_2['Work'].replace({'Private': 1, 'Self-employed':2, 'children':3, 'Govt_job':4, 'Never_worked': 5 }, inplace=True)
df_2['Residence'].replace({'Urban': 1, 'Rural':0}, inplace=True)
df_2['Smoking'].replace({'formerly smoked': 1, 'never smoked':2, 'smokes':3, 'Unknown':4}, inplace=True)
df_2['Gender'].replace({'Male': 1, 'Female':0}, inplace=True)
df_2


# Nuevo tamaño de la muestra
df_2
print(df_2.shape)
print(pd.value_counts(df_2['Stroke'], sort = True))
# Gráfico de distribución de la variable dependiente
count_classes = pd.value_counts(df_2['Stroke'], sort = True, normalize=True)
count_classes *= 100




# Calcular la frecuencia de cada categoría de Stroke en porcentajes
count_classes = pd.value_counts(df_2['Stroke'], sort=True, normalize=True)
count_classes *= 100

# Crear una figura con un tamaño de 10 pulgadas de ancho y 8 pulgadas de alto
plt.figure(figsize=(15, 8))

# Crear el gráfico de barras con porcentajes
ax = count_classes.plot(kind='bar', rot=0)

# Añadir etiquetas con porcentajes a cada barra del gráfico
for i in ax.patches:
    ax.text(i.get_x() + 0.1, i.get_height() + 0.5, str(round(i.get_height(), 2)) + '%', fontsize=10, color='black')

# Ajustar las etiquetas del eje x para que coincidan con las categorías de Stroke
plt.xticks(range(2))

# Añadir un título al gráfico
plt.title("Frecuencia variable Stroke")

# Añadir una etiqueta al eje x
plt.xlabel("Stroke")

# Añadir una etiqueta al eje y
plt.ylabel("Porcentaje de la Observación")

# Mostrar el gráfico
plt.show()

# Separacion de la base de prueba entrenamiento y validación
#definimos nuestras etiquetas y features
y = df_2['Stroke']
X = df_2.drop(columns=['Stroke'], axis=1)

# Dividimos en conjuntos de entrenamiento y prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Dividimos el conjunto de entrenamiento en conjunto de entrenamiento y validación (60/40)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Función para representar el forward selection
def forward_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out = 0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded,dtype='float64')
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        min_pval = new_pval.min()
        if min_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(new_pval.index[best_feature])
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, min_pval))
        if not changed:
            break
    return included

model = sm.Logit(y_train, sm.add_constant(X_train['Hypertension'])).fit(disp=0)
selected_features = forward_selection(X_train, y_train, initial_list=['Hypertension'], verbose=True)
selected_features
model = sm.Logit(y_val, sm.add_constant(X_val['Hypertension'])).fit(disp=0)
selected_features = forward_selection(X_val, y_val, initial_list=['Hypertension'], verbose=True)
selected_features
# Estimación del modelo de regresión logística usando las bases de entrenamiento
X_train_final = X_train[selected_features]
model_final = sm.Logit(y_train, sm.add_constant(X_train_final)).fit(disp=0)
model_final.summary()
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Función para calcular y mostrar los resultados
def mostrar_resultados(y_true, y_pred, y_prob):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print (classification_report(y_true, y_pred))
    
    print("Matriz de confusión:")
    print(conf_matrix)

    acc = accuracy_score(y_true, y_pred)
    print("Exactitud:", acc)

    prec = precision_score(y_true, y_pred)
    print("Precisión:", prec)

    rec = recall_score(y_true, y_pred)
    print("Sensibilidad:", rec)

    spec = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    print("Especificidad:", spec)

    f1 = f1_score(y_true, y_pred)
    print("Puntuación F1:", f1)

    auc = roc_auc_score(y_true, y_prob)
    print("AUC:", auc)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label="Curva ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Modelo aleatorio")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()



# Regresión Logística sin balancear la base

# #definimos nuestras etiquetas y features
# y = df_2['Stroke']
# X = df_2[selected_features]


# # Dividimos en conjuntos de entrenamiento y prueba (70/30)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# # Dividimos el conjunto de entrenamiento en conjunto de entrenamiento y validación (60/40)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# #creamos una función que crea el modelo logistico
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg")
    clf_base.fit(X_train, y_train)
    # X_train_sm = sm.add_constant(X_train)
    # model = sm.Logit(y_train, X_train_sm)
    # results = model.fit()
    # print(results.summary())
    return clf_base

# #ejecutamos el modelo "tal cual"
model = run_model(X_train, X_test, y_train, y_test)


# #definimos funciona para mostrar los resultados, creando la matriz de confusión
# def mostrar_resultados(y_test, pred_y):
#     conf_matrix = confusion_matrix(y_test, pred_y)
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt="d");
#     plt.title("Confusion matrix")
#     plt.ylabel('True class')
#     plt.xlabel('Predicted class')
#     plt.show()
#     print (classification_report(y_test, pred_y))

pred_y = model.predict(X_test)
#mostrar_resultados(y_test, pred_y)




# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
# ESTRATEGIA 1: PENALIZACION PARA COMPENSAR

def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    clf.fit(X_train, y_train)
    X_train_sm = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_train_sm)
    results = model.fit()
    print(results.summary())
    return clf
 
model = run_model_balanced(X_train, X_test, y_train, y_test)
pred_y = model.predict(X_test)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
# Probando valores desde el 1 al 14 vecinos 
results = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    results.append({
        'k': k,
        'accuracy': accuracy_score(y_test, knn.predict(X_test))
    })

# Convert results to a pandas data frame
results = pd.DataFrame(results)
print(results)

# Which value of k should be selected?
us = NearMiss(sampling_strategy=0.5, n_neighbors=14, version=2)
X_train_res, y_train_res = us.fit_resample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))

model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
os =  RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution labels after resampling {}".format(Counter(y_train_res)))
 
model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
import scipy.stats as stats
import numpy as np

y_train 
y_train_res 

_, p_val, _, _ = stats.chi2_contingency([np.bincount(y_train), np.bincount(y_train_res)])
print("p-value: ", p_val)



os_us = SMOTETomek(sampling_strategy=0.5)
X_train_res, y_train_res = os_us.fit_resample(X_train, y_train)

print ("Distribution before resampling {}".format(Counter(y_train)))
print ("Distribution after resampling {}".format(Counter(y_train_res)))

model = run_model(X_train_res, X_test, y_train_res, y_test)
pred_y = model.predict(X_test)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

#Train the classifier.
model = run_model(X_train_res, X_test, y_train_res, y_test)

bbc.fit(X_train, y_train)
pred_y = bbc.predict(X_test)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = bbc.predict(X_train)
train_prob = bbc.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = bbc.predict(X_val)
val_prob = bbc.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = bbc.predict(X_test)
test_prob = bbc.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)

# Print the class distribution before applying Tomek Links
print("Class distribution before Tomek Links:", Counter(y_train))

# Identify Tomek Links in the training data
tl = TomekLinks()
X_train_res, y_train_res = tl.fit_resample(X_train, y_train)

model = run_model(X_train_res, X_test, y_train_res, y_test)

# Print the class distribution after applying Tomek Links
print("Class distribution after Tomek Links:", Counter(y_train_res))

# Train a decision tree classifier on the under-sampled training data
clf = DecisionTreeClassifier(random_state=10)
clf.fit(X_train_res, y_train_res)

# Use the trained classifier to make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
# Calcular los resultados para el conjunto de entrenamiento
train_pred = clf.predict(X_train)
train_prob = clf.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = clf.predict(X_val)
val_prob = clf.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = clf.predict(X_test)
test_prob = clf.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# # cargar los datos del hospital desde un archivo CSV
# data = pd.read_csv("datos_hospital.csv")

# # separar la variable objetivo
# y = data["Stroke"]
# X = data.drop(["Stroke"], axis=1)

# # dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os =  RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X_train, y_train)

# definir el espacio de búsqueda de hiperparámetros
param_distributions = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# crear el modelo RandomForest y definir el método Random Search
rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_distributions, n_iter=20, cv=5, n_jobs=-1)

# ajustar el modelo al conjunto de entrenamiento
random_search.fit(X_train_res, y_train_res)

# evaluar el modelo en el conjunto de prueba
y_pred = random_search.predict(X_test)

# imprimir la matriz de confusión
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# imprimir el reporte de clasificación
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# imprimir los mejores hiperparámetros encontrados
print("\nMejores hiperparámetros:", random_search.best_params_)


# Calcular los resultados para el conjunto de entrenamiento
train_pred = random_search.predict(X_train)
train_prob = random_search.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = random_search.predict(X_val)
val_prob = random_search.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = random_search.predict(X_test)
test_prob = random_search.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)
pip install scikit-optimize
# Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_score

# # Cargar datos
# data = pd.read_csv("hospital.csv")

# # Preprocesamiento de datos
# le = LabelEncoder()
# for col in data.select_dtypes(include='object'):
#     data[col] = le.fit_transform(data[col].astype(str))
# data = data.dropna()
# X = data.drop("Stroke", axis=1)
# y = data["Stroke"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir espacio de búsqueda para hiperparámetros
space = [Integer(10, 150, name='n_estimators'),
         Integer(1, 20, name='max_depth'),
         Categorical(['gini', 'entropy'], name='criterion'),
         Real(0.001, 0.5, name='min_samples_split'),
         Real(0.001, 0.5, name='min_samples_leaf')]

# Definir función objetivo para la optimización de hiperparámetros
def objective(values):
    n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf = values
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   criterion=criterion,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)
    score = cross_val_score(model, X_train_res, y_train_res, cv=5).mean()
    return 1 - score

# Ejecutar optimización de hiperparámetros con Bayesian Optimization
result = gp_minimize(objective, space, random_state=42)

# Entrenar modelo con los mejores hiperparámetros encontrados
best_params = {'n_estimators': result.x[0],
               'max_depth': result.x[1],
               'criterion': result.x[2],
               'min_samples_split': result.x[3],
               'min_samples_leaf': result.x[4]}
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluar modelo con la matriz de confusión
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", cm)

# Imprimir reporte de clasificación
cr = classification_report(y_test, y_pred)
print("Reporte de Clasificación:\n", cr)

# Calcular los resultados para el conjunto de entrenamiento
train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]
print("Resultados del conjunto de entrenamiento:")
mostrar_resultados(y_train, train_pred, train_prob)

# Calcular los resultados para el conjunto de validación
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print("Resultados del conjunto de validación:")
mostrar_resultados(y_val, val_pred, val_prob)

# Calcular los resultados para el conjunto de prueba
test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
print("Resultados del conjunto de prueba:")
mostrar_resultados(y_test, test_pred, test_prob)

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# # Cargar los datos
# data = pd.read_excel("hospital_data.xlsx")

# # Definir la variable objetivo y las variables explicativas
# y = data["Stroke"]
# X = data.drop("Stroke", axis=1)

# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os =  RandomOverSampler(sampling_strategy=0.5)
X_train_res, y_train_res = os.fit_resample(X_train, y_train)


# Crear la matriz de datos XGBoost a partir de los datos de entrenamiento y prueba
dtrain = xgb.DMatrix(X_train_res, label=y_train_res)
dtest = xgb.DMatrix(X_test, label=y_test)
dval = xgb.DMatrix(X_val, label=y_val)

# Definir los hiperparámetros para el modelo XGBoost
params = {
    "max_depth": 3,
    "eta": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

# Entrenar el modelo XGBoost con los datos de entrenamiento y los hiperparámetros definidos
model = xgb.train(params, dtrain)

# Realizar las predicciones para los datos de entrenamiento y prueba
y_train_pred = model.predict(dtrain)
y_test_pred = model.predict(xgb.DMatrix(X_test))
y_val_pred = model.predict(dval)

train_prob = model.predict(dtrain)
val_prob = model.predict(dval)
test_prob = model.predict(xgb.DMatrix(X_test))

# Convertir las probabilidades en etiquetas binarias (0 o 1) usando un umbral de 0.5
y_train_pred_binary = [1 if p > 0.5 else 0 for p in y_train_pred]
y_test_pred_binary = [1 if p > 0.5 else 0 for p in y_test_pred]
y_val_pred_binary = [1 if p > 0.5 else 0 for p in y_val_pred]

# Calcular la precisión y la matriz de confusión para los datos de entrenamiento y prueba
train_accuracy = accuracy_score(y_train_res, y_train_pred_binary)
test_accuracy = accuracy_score(y_test, y_test_pred_binary)
train_cm = confusion_matrix(y_train_res, y_train_pred_binary)
test_cm = confusion_matrix(y_test, y_test_pred_binary)
val_accuracy = accuracy_score(y_val, y_val_pred_binary)
val_cm = confusion_matrix(y_val, y_val_pred_binary)

# Imprimir el reporte de clasificación y las matrices de confusión para los datos de entrenamiento y prueba
print("Training Classification Report:")
mostrar_resultados(y_train_res, y_train_pred_binary, train_prob)


# Calcular los resultados para el conjunto de validación
print("Valid Classification Report:")
mostrar_resultados(y_val, y_val_pred_binary, val_prob)


# Calcular los resultados para el conjunto de validación
print("Test Classification Report:")
mostrar_resultados(y_test, y_test_pred_binary, test_prob)


# # Calcular los resultados para el conjunto de entrenamiento
#train_pred = random_search.predict(X_train)
#train_prob = random_search.predict_proba(X_train)[:, 1]
# print("Resultados del conjunto de entrenamiento:")
#mostrar_resultados(y_train, train_pred, train_prob)

# # Calcular los resultados para el conjunto de validación
# val_pred = random_search.predict(X_val)
# val_prob = random_search.predict_proba(X_val)[:, 1]
# print("Resultados del conjunto de validación:")
# mostrar_resultados(y_val, val_pred, val_prob)

# # Calcular los resultados para el conjunto de prueba
# test_pred = random_search.predict(X_test)
# test_prob = random_search.predict_proba(X_test)[:, 1]
# print("Resultados del conjunto de prueba:")
# mostrar_resultados(y_test, test_pred, test_prob)


print("Valid Confusion Matrix:")
print(test_cm)
print(classification_report(y_val, y_val_pred_binary))
print("Valid Confusion Matrix:")
print(test_cm)
# REFERENCIAS
# Bhattacharyya, Jayita. “Handling Imbalanced Datasets: A Guide with Hands-on Implementation.” Analytics India Magazine, 21 Oct. 2020, analyticsindiamag.com/handling-imbalanced-datasets-a-guide-with-hands-on-implementation/.
# Khanam, Jobeda Jamal, and Simon Y. Foo. “A Comparison of Machine Learning Algorithms for Diabetes Prediction.” ICT Express, Feb. 2021, https://doi.org/10.1016/j.icte.2021.02.004.
# Team, Towards AI. “Important Techniques to Handle Imbalanced Data in Machine Learning… – towards AI.” Towardsai.net, 19 Sept. 2022, towardsai.net/p/l/important-techniques-to-handle-imbalanced-data-in-machine-learning-python. Accessed 11 Apr. 2023.
# Brownlee, J. (2020). Imbalanced classification with Python: better metrics, balance skewed classes, cost-sensitive learning. Machine Learning Mastery.https://books.google.com.ec/books?hl=es&lr=&id=uAPuDwAAQBAJ&oi=fnd&pg=PP1&dq=Dealing+with+Imbalanced+Datasets+in+Machine+Learning+de+Brownlee+(2020)&ots=Cl8FucfTuY&sig=RO3SyhfjsE46o0r6Rz6PIhunskc#v=onepage&q=Dealing%20with%20Imbalanced%20Datasets%20in%20Machine%20Learning%20de%20Brownlee%20(2020)&f=false
# Tsai, Chih-Fong, et al. “Under-Sampling Class Imbalanced Datasets by Combining Clustering Analysis and Instance Selection.” Information Sciences, vol. 477, 1 Mar. 2019, pp. 47–54, www.sciencedirect.com/science/article/abs/pii/S0020025518308478, https://doi.org/10.1016/j.ins.2018.10.029
# Machine Learning, Blog. “Clasificación Con Datos Desbalanceados | Aprende Machine Learning.” Www.aprendemachinelearning.com, 9 May 2019, www.aprendemachinelearning.com/clasificacion-con-datos-desbalanceados/.
# Morales Oñate, Víctor Hugo, et al. “SMOTEMD: Un Algoritmo de Balanceo de Datos Mixtos Para Big Data En R.” Dspace.espoch.edu.ec, 24 Apr. 2020, dspace.espoch.edu.ec/handle/123456789/14586. Accessed 11 Apr. 2023.

# Niyogisubizo, Jovial, et al. “Predicting Student’s Dropout in University Classes Using Two-Layer Ensemble Machine Learning Approach: A Novel Stacked Generalization.” Computers and Education: Artificial Intelligence, vol. 3, 2022, p. 100066, https://doi.org/10.1016/j.caeai.2022.100066.
# Vasudha, Vashisht. Learning from Imbalanced Data. 9 Sept. 2009, www.academia.edu/29164932/Learning_from_Imbalanced_Data. Accessed Apr. 8AD.
