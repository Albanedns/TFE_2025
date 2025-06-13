**preprocessing.py** contient le code ayant permis de créer le fichier CSV avec les informations sur les participants ainsi que les métriques calculées à partir des fichiers de logs.
**errors.csv** est le résultat généré par preprocessing.py.

## Analyse Statistique ##

**landmark_usage.py** contient le code utilisé pour étudier l'utilisation du landmark (temps durant lequel le phare apparait sur l'écran). 



## Machine Learning ## 

**random_forest.py** contient le code utilisé pour appliquer un classificateur Random Forest sur le dataset afin de prédire le statut APOE, avec ou sans biais (en modifiant quelques lignes dans le script).

**LSTM_FCN.py** contient le code utilisé pour entraîner un classificateur LSTM-FCN. Les paramètres du modèle ont été modifiés directement dans le script.

**AutoEncodeur_LSTM.py** contient le code utilisé pour entrainer l'autoencodeur LSTM. Les paramètres du modèle ont été modifiés directement dans le script.
