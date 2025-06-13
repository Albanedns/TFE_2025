**preprocessing.py** contient le code ayant permis de créer le fichier CSV avec les informations sur les participants ainsi que les métriques calculées à partir des logfiles.

**errors.csv** est le résultat généré par preprocessing.py.

## Analyse Statistique ##

**mixed_linear.py** contient le code utilisé pour examiner l'effet de l’âge et du gène APOE sur l'erreur de dépot, l'erreur de rotation et l'erreur de distance.

**view_of_basket.py** contient le code utilisé pour étudier l’effet de la vue initiale du panier et de la distance correcte sur la distance parcourue par le participant lors de la phase incoming.
Par défaut, le code examine l'effet de la distance correcte sur la variable incoming_distance. Pour analyser l'effet de la vue initiale du panier, il suffit de remplacer "incoming_distance" par "start_basket_distance" dans le script.

**landmark_usage.py** contient le code permettant d’analyser l’utilisation du landmark, c’est-à-dire le temps pendant lequel le phare apparaît à l’écran.


**new_metrics.py** contient le code utilisé pour examiner l'effet de l’âge et du gène APOE sur plusieurs nouvelles métriques calculées :
- le temps total,
- le temps de la phase incoming,
- la vitesse moyenne pendant cette phase,
- le path length ratio (version 1 et 2),
- la distance absolue moyenne (version 1 et 2).

Actuellement, le script est configuré pour analyser l'effet de l’âge et de l’APOE sur le path length ratio version 1 dans la condition Pure Path Integration.


## Machine Learning ## 

**random_forest.py** contient le code utilisé pour appliquer un classificateur Random Forest sur le dataset afin de prédire le statut APOE, avec ou sans biais (en modifiant quelques lignes dans le script).

**LSTM_FCN.py** contient le code utilisé pour entraîner un classificateur LSTM-FCN. Les paramètres du modèle ont été modifiés directement dans le script.

**AutoEncodeur_LSTM.py** contient le code utilisé pour entrainer l'autoencodeur LSTM. Les paramètres du modèle ont été modifiés directement dans le script.
