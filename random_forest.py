import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, auc, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn import tree
import seaborn as sns
import shap
from fairlearn.metrics import MetricFrame


#chemin vers le fichier CSV qui contient les différents essais avec les métrics calculées
file_path = "errors.csv" 
df = pd.read_csv(file_path)

#création de la variable education_group qui regroupe les niveaux d'éducation en 3 catégories (low, medium, high)
df['education_group'] = pd.qcut(
    df['Education'],
    q=3,
    labels=['low', 'medium', 'high']
)

#décommenter si on veut l'utiliser comme label 
#df['MMSE2'] = np.where(df['MMSE'] >= 27, 1, 0)

df = df.drop(columns=["trial_number", "block_number","age",'Participant',"VersionAppleGame","Education"])

df = df.replace(-999, np.nan)  
df = df.dropna() 

#creaction de X et y (le label)
X = df.drop(columns=["APOE"])
y = df["APOE"]


#Encodage des variables categorielles
label_encoders = {}
categorical_cols = ["Gender", "AgeGroup", "education_group"]

categories_ordre = {}
categories_ordre["AgeGroup"] = [["<=50", "51-60", "61-70", ">70"]]
categories_ordre["education_group"] =[["low", "medium", "high"]]
for col in categorical_cols:
    if (col == "AgeGroup" or col =="education_group"):
        le = OrdinalEncoder(categories=categories_ordre[col])
    else :
        le = OneHotEncoder(sparse_output=False, drop='first')
    X[col] = le.fit_transform(X[[col]])
    label_encoders[col] = le  

#print(X["Gender"].unique())
#print(X["AgeGroup"].unique())

np.random.seed(42)
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist() 
#Splitting le dataset en training set et test set 
X_train, X_test, y_train, y_test, A_train, A_test= train_test_split(X, y,X["education_group"],test_size=0.2, random_state=123, stratify=y)

#Affichage de la distribution de l'APOE par groupe d'education
df2 = pd.DataFrame(X_test)
df2['APOE'] = y_test
proportions = df2.groupby('education_group')['APOE'].value_counts(normalize=True).rename('Class distribution').reset_index()
proportions['APOE'] = proportions['APOE'].replace({0.0: 'ε4 non-carriers', 1.0: 'ε4 carriers'})

sns.barplot(data=proportions, x='education_group', y='Class distribution', hue='APOE')
plt.title("Class distribution by Education Group")
plt.legend(title="APOE status")
plt.show()

#Standardisation des données
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


#Grid Search (ici j'ai gardé que les parametres du meilleur modèle)
param_grid = {
    'n_estimators': [ 200],  
    'max_depth': [30],  
    'min_samples_split': [5], 
    'min_samples_leaf': [2], 
    'class_weight': ['balanced'], 
    'max_samples': [1.0],
    'max_features': [0.7]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy', verbose=2)

grid_search.fit(X_train, y_train)

#Print les meilleurs parametres
print("Best parameters from GridSearchCV:", grid_search.best_params_)


best_rf = grid_search.best_estimator_

#Test du random forest sur le test set 
y_pred = best_rf.predict(X_test)

#accuracy par group d'education
mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
print(mf.by_group)
print(y_pred)

mf.by_group.sort_index(inplace=True)  # Pour que l'axe x soit ordonné

plt.figure(figsize=(10, 5))
plt.bar(mf.by_group.index.astype(str), mf.by_group.values)
plt.xlabel("Years of Education")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score by Education Level")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Print des résultats sur le test set 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

#Importance des features
importances = pd.Series(best_rf.feature_importances_, index=df.drop(columns=["APOE"]).columns)
print("Feature Importance:\n", importances.sort_values(ascending=False))






