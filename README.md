**preprocessing.py** contient le code ayant permis de créer le fichier CSV avec les informations sur les participants ainsi que les métriques calculées à partir des fichiers de logs.

**error.csv** est le résultat généré par preprocessing.py.

**random_forest.py** contient le code utilisé pour appliquer un classificateur Random Forest sur le dataset afin de prédire le statut APOE, avec ou sans biais (en modifiant quelques lignes dans le script).

**LSTM_FCN.py** contient le code utilisé pour entraîner un classificateur LSTM-FCN. Les paramètres du modèle ont été modifiés directement dans le script.
