import scipy.io
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('data_TFE_24_25/info_participant_anonyme.xlsx')
logfile_dir = 'data_TFE_24_25/logfiles/'


"""
Calcul de la drop error pour chaque logfile
"""
drop_errors = []
trial_numbers = []
participant_numbers = []
for subdir, dirs, files in os.walk(logfile_dir):
    participant_number = os.path.basename(subdir)
    for file in files:
        if file.endswith('_data.mat'): 
            mat_path = os.path.join(subdir, file)
            df1 = scipy.io.loadmat(mat_path)
            data = df1['data']
            

            filtered_data = data[(data[:, 10] != -999.0) & (data[:, 11] != -999.0)]
            
            unique_trials = np.unique(filtered_data[:, 8])
            
            for trial in unique_trials:
                trial_data = filtered_data[filtered_data[:, 8] == trial]
                trial_phase4_data = trial_data[trial_data[:, 9] == 4]
                if trial_phase4_data.size > 0:
                    player_x = trial_phase4_data[-1, 1]
                    player_y = trial_phase4_data[-1, 2]
        

                    correct_x = trial_phase4_data[-1, 10]
                    correct_y = trial_phase4_data[-1, 11]
        
                    drop_error = np.sqrt((player_x - correct_x)**2 + (player_y - correct_y)**2)

                    drop_errors.append(drop_error)
                    trial_numbers.append(int(trial))
                    participant_numbers.append(participant_number) 
                else: 
                    drop_errors.append(np.nan)
                    trial_numbers.append(int(trial))
                    participant_numbers.append(participant_number) 


df_result = pd.DataFrame({
    'participant_number': participant_numbers,
    'trial_number': trial_numbers,
    'drop_error': drop_errors
})

"""
ajout de l'age
"""
df_result['participant_number_clean'] = df_result['participant_number'].str.split('_').str[0]
df['numAppleGame'] = df['numAppleGame'].apply(lambda x: str(x).zfill(3))

df_result = pd.merge(df_result, df[['numAppleGame', 'age']], 
                     left_on='participant_number_clean', right_on='numAppleGame', 
                     how='left')


df_result.drop(columns=['participant_number_clean'], inplace=True)
df_result.drop(columns=['numAppleGame'], inplace=True)

"""
Calcul de toutes les autres métriques
"""
def path_length_ratio(trajectory, objective=[]):
    """
    Calcule le Path Length Ratio d'une trajectoire donnée.
    
    :param trajectory: Liste ou tableau numpy de points (x, y)
    :param objective: None (pour la version 1) or True Basket Location (pour la version 2)
    :return: Path Length Ratio (PLR)
    """
    trajectory = np.array(trajectory)
    
    path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    start, end = trajectory[0], trajectory[-1]
    if len(objective) > 0:
        start, end = trajectory[0], objective
    euclidean_distance = np.linalg.norm(end - start)
    
    return path_length / euclidean_distance if euclidean_distance != 0 else np.nan

def mean_absolute_distance_to_line(trajectory, objective = []):
    """
    Calcule la Mean Absolute Distance (MAD) entre une trajectoire et une ligne droite entre le premier et le dernier point.

    :param trajectory: Liste ou tableau numpy de points (x, y)
    :param objective: None (pour la version 1) or True Basket Location (pour la version 2)
    :return: MAD (Mean Absolute Distance)
    """
    trajectory = np.array(trajectory)
    x, y = trajectory[:, 0], trajectory[:, 1]  

    x1, y1 = x[0], y[0]
    if len(objective) > 0:
        xn, yn = objective 
    else:
        xn, yn = x[-1], y[-1]

    A = y1 - yn
    B = xn - x1
    C = x1 * yn - xn * y1

    distances = np.abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

    MAD = np.mean(distances)
    
    return MAD

x_phare, y_phare= 1600,800 #position du phare
FoV = 90 
rotation_errors=[] 
distance_errors=[]
landmark=[] # 1 si landmark 0 sinon
temps_de_vision = [] # temps durant lequel le phare est visible
response_distances=[] # distance parcourue par le participant lors de l'incoming phase
incoming_distances=[] # distance qu'il aurait fallu parcourir lors de l'incoming phase
start_basket_distances=[] #distance du panier lors de la start phase
path_length_ratios=[] #PLR de l'incoming phase version 1
path_length_ratios2=[] #PLR de l'incoming phase version 2
time_tasks=[] #temps pour effectuer toute la tâche (1 essai)
times_phase4=[] #temps pour effectuer l'incoming phase
mads= [] #MAD de l'incoming phase version 1
mads2=[] #MAD de l'incoming phase version 2
mean_speeds=[] #vitesse moyenne de l'incoming phase
for subdir, dirs, files in os.walk(logfile_dir):
    participant_number = os.path.basename(subdir)
    mat_file_exist= False
    for file in files:
        if file.endswith('_data.mat'): 
            mat_file_exist= True
            mat_path = os.path.join(subdir, file)
            df2 = scipy.io.loadmat(mat_path)
            data = df2['data']
            

            filtered_data = data[(data[:, 10] != -999.0) & (data[:, 11] != -999.0)]
            
            unique_trials = np.unique(filtered_data[:, 8])
            
            for trial in unique_trials:
                trial_data = filtered_data[filtered_data[:, 8] == trial]
                trial_phase1_data = trial_data[trial_data[:,9]==1]
                trial_phase2_data = trial_data[trial_data[:,9]==2]
                trial_phase3_data = trial_data[trial_data[:, 9] == 3] 
                trial_phase4_data = trial_data[trial_data[:, 9] == 4] 
                if(trial_phase1_data.shape[0]>0):
                    time1 = trial_phase1_data[-1,0] - trial_phase1_data[0,0]
                    
                distances=[]
                if (trial_phase2_data.shape[0]>0):
                    
                    objectives_phase2 = trial_phase2_data[:, [10, 11]] 
                    point1 = trial_phase2_data[0, [1, 2]]
                    distances.append(np.linalg.norm(objectives_phase2- point1))
                    unique_objectives = np.unique(objectives_phase2, axis=0)  
    
                    for objective in unique_objectives:
        
                        start_indices = np.where(np.all(trial_phase2_data[:, [10, 11]] == objective, axis=1))[0]
                        if len(start_indices) > 0:
                            start_time = trial_phase2_data[start_indices[0], 0] 
        
                        end_indices = np.where(np.all(trial_phase2_data[:, [10, 11]] == objective, axis=1))[0]
                        if len(end_indices) > 0:
                            end_time = trial_phase2_data[end_indices[-1], 0]  
      
                        
                if (trial_phase3_data.shape[0]>0): 
                    objectives_phase3 = trial_phase3_data[:, [10, 11]] 
                    point1 = trial_phase3_data[0, [1, 2]]
                    distances.append(np.linalg.norm(objectives_phase3- point1))
                    time3=  trial_phase3_data[-1,0] - trial_phase3_data[0,0] 
                    
                if trial_phase4_data.shape[0]>0:
                    time = trial_phase4_data[-1,0] - trial_phase1_data[0,0]
                    times_phase4.append(trial_phase4_data[-1,0]-trial_phase4_data[0,0])
                    distances = np.sqrt(np.diff(trial_phase4_data[:,1])**2 + np.diff(trial_phase4_data[:,2])**2)
        
                    # Calculer la distance totale
                    total_distance = np.sum(distances)

                    # Calculer la vitesse moyenne (distance totale / temps total)
                    mean_speed = total_distance / time if time>0 else 0
                    mean_speeds.append(mean_speed)
                else:
                    time = np.nan
                    times_phase4.append(np.nan)
                    mean_speeds.append(np.nan)
                time_tasks.append(time)
                if trial_phase4_data.shape[0] > 1:  # Vérifie s'il y a au moins 2 points
                    objectives_phase4 = trial_phase4_data[0, [10, 11]] 
                    plr =path_length_ratio(np.column_stack((trial_phase4_data[:,1], trial_phase4_data[:,2])))
                    plr2 = path_length_ratio(np.column_stack((trial_phase4_data[:,1], trial_phase4_data[:,2])), objectives_phase4)
                    mad = mean_absolute_distance_to_line(np.column_stack((trial_phase4_data[:,1], trial_phase4_data[:,2])))
                    mad2 = mean_absolute_distance_to_line(np.column_stack((trial_phase4_data[:,1], trial_phase4_data[:,2])), objectives_phase4)
                else:
                    plr = np.nan
                    plr2= np.nan
                    mad = np.nan
                    mad2=np.nan
                path_length_ratios.append(plr)
                path_length_ratios2.append(plr2)
                mads.append(mad)
                mads2.append(mad2)
                start_of_phases = trial_data[trial_data[:, 9] == 1]
                start_position_x, start_position_y = start_of_phases[0, 1], start_of_phases[0, 2]
                phasetype_1 = trial_data[trial_data[:, 9]== 1]
                basket_x, basket_y = phasetype_1[0, 10], phasetype_1[0, 11]
                start_basket_distance = np.sqrt((basket_x - start_position_x)**2 + (basket_y - start_position_y)**2)
                start_basket_distances.append(start_basket_distance)
                if (trial_data[:, 7]==3).all():
                    landmark.append(1)
                    vision_time = 0

                    for i in range(1, len(trial_data)):  
                        time_prev, x_prev, y_prev, yaw_prev = trial_data[i - 1, [0, 1, 2, 4]]
                        time_curr, x_curr, y_curr, yaw_curr = trial_data[i, [0, 1, 2, 4]]

                        # Vecteur direction du joueur
                        dir_joueur = np.array([np.cos(np.radians(yaw_curr)), np.sin(np.radians(yaw_curr))])

                        # Vecteur joueur → phare
                        vec_joueur_phare = np.array([x_phare - x_curr, y_phare - y_curr])
                        norm_phare = np.linalg.norm(vec_joueur_phare)

                        if norm_phare > 0:
                            vec_joueur_phare /= norm_phare  # Normaliser

                        # Angle entre la direction du joueur et le phare
                            cos_theta = np.clip(np.dot(dir_joueur, vec_joueur_phare), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_theta))

                            # Vérifier si le phare est dans le champ de vision
                            if angle_deg <= (FoV / 2):
                                vision_time += (time_curr - time_prev)  # Ajouter le temps écoulé
                
                    temps_de_vision.append(vision_time)
                elif (trial_data[:, 7]==1).all():
                    landmark.append(0)
                    temps_de_vision.append(0)
                if trial_phase3_data.size > 0 and trial_phase4_data.size > 0:
                    
                    drop_x = trial_phase3_data[-1, 1] 
                    drop_y = trial_phase3_data[-1, 2] 
                    
                    player_x = trial_phase4_data[-1, 1] 
                    player_y = trial_phase4_data[-1, 2] 
                    
                    correct_x = trial_phase4_data[-1, 10]  
                    correct_y = trial_phase4_data[-1, 11]  
                    yaw = np.radians(trial_phase4_data[-1,4])
                   
                    vector_drop_to_player = [player_x - drop_x, player_y - drop_y]
                    vector_drop_to_basket = [correct_x - drop_x, correct_y - drop_y]

                    response_distance = np.linalg.norm(vector_drop_to_player)
                    incoming_distance = np.linalg.norm(vector_drop_to_basket)
                    dot_product = np.dot(vector_drop_to_player, vector_drop_to_basket)
                    
                    
                    if response_distance == 0:
                        
                        angle_between_vectors_degrees = np.nan  # Impossible de calculer l'angle

                    else:
                        # Angle entre le vecteur drop → player et drop → basket
                        dot_product = np.dot(vector_drop_to_player, vector_drop_to_basket)
                        magnitude_player = np.linalg.norm(vector_drop_to_player)

                        if magnitude_player > 0 and incoming_distance > 0:
                            angle_between_vectors = np.arccos(np.clip(dot_product / (magnitude_player * incoming_distance), -1.0, 1.0))
                            angle_between_vectors_degrees = np.degrees(angle_between_vectors)
                        else:
                            angle_between_vectors_degrees = np.nan
                    
                    rotation_error = min(angle_between_vectors_degrees, 360 - angle_between_vectors_degrees)
                    rotation_errors.append(rotation_error)

                    
                    incoming_distance = np.sqrt((drop_x - correct_x)**2 + (drop_y - correct_y)**2)
                    
                    response_distance = np.sqrt((player_x - drop_x)**2 + (player_y - drop_y)**2)
                    
                    
                    distance_error = np.abs(incoming_distance - response_distance)
                    distance_errors.append(distance_error)
                    response_distances.append(response_distance)
                    incoming_distances.append(incoming_distance)


                else:
                    rotation_errors.append(np.nan)
                    distance_errors.append(np.nan)
                    response_distances.append(np.nan)
                    incoming_distances.append(np.nan)
    

df_result['rotation_error'] = rotation_errors  
df_result['distance_error'] = distance_errors
df_result['response_distance']= response_distances
df_result['incoming_distance']= incoming_distances
df_result['start_basket_distance']= start_basket_distances
df_result['landmark']= landmark
df_result['aide_phare']= temps_de_vision
df_result['path_length_ratio_1']=path_length_ratios
df_result['path_length_ratio_2']=path_length_ratios2
df_result['time']= time_tasks
df_result['time_phase4']= times_phase4
df_result['mean_abs_value']=mads
df_result['mean_abs_value2']=mads2
df_result['mean_speed']=mean_speeds

"""
Ajout du nombre d'arbre dans un essai (1, 2 ou 3)
"""
tree_counts = []  

for subdir, dirs, files in os.walk(logfile_dir):
    participant_number = os.path.basename(subdir)
    for file in files:
        if file.endswith('_data.mat'): 
            mat_path = os.path.join(subdir, file)
            
            df2 = scipy.io.loadmat(mat_path)
            data = df2['data']
            
            filtered_data = data[(data[:, 10] != -999.0) & (data[:, 11] != -999.0)]
            
            unique_trials = np.unique(filtered_data[:, 8])  
            
            for trial in unique_trials:
                trial_data = filtered_data[filtered_data[:, 8] == trial]
                last_x, last_y = None, None
                tree_count = 0
                
                for row in trial_data:
                    current_x, current_y = row[10], row[11]  
                    phasetype = row[9]  
                    
                      
                    if (phasetype == 2 or phasetype == 3) and (last_x != current_x or last_y != current_y):
                        tree_count += 1 
                        last_x, last_y = current_x, current_y  
                tree_counts.append(tree_count) 

df_result['tree_count'] = tree_counts 

"""
Ajout du nombre d'année d'étude
"""
df_result['participant_number_clean'] = df_result['participant_number'].str.split('_').str[0]
df['numAppleGame'] = df['numAppleGame'].apply(lambda x: str(x).zfill(3))
df_result = pd.merge(df_result, df[['numAppleGame', 'AnneeEtude']], 
                     left_on='participant_number_clean', right_on='numAppleGame', 
                     how='left')


df_result.drop(columns=['participant_number_clean'], inplace=True)
df_result.drop(columns=['numAppleGame'], inplace=True)

"""
Ajout du nombre d'heure de jeu video
"""
df_result['participant_number_clean'] = df_result['participant_number'].str.split('_').str[0]
df['numAppleGame'] = df['numAppleGame'].apply(lambda x: str(x).zfill(3))
df_result = pd.merge(df_result, df[['numAppleGame', 'Nbre_heure_jeux_videos']], 
                     left_on='participant_number_clean', right_on='numAppleGame', 
                     how='left')


df_result.drop(columns=['participant_number_clean'], inplace=True)
df_result.drop(columns=['numAppleGame'], inplace=True)

"""
Ajout de la version de l'Apple Game
"""
df_result['participant_number_clean'] = df_result['participant_number'].str.split('_').str[0]
df['numAppleGame'] = df['numAppleGame'].apply(lambda x: str(x).zfill(3))
df_result = pd.merge(df_result, df[['numAppleGame', 'VersionAppleGame']], 
                     left_on='participant_number_clean', right_on='numAppleGame', 
                     how='left')


df_result.drop(columns=['participant_number_clean'], inplace=True)
df_result.drop(columns=['numAppleGame'], inplace=True)

"""
Ajout du Genre
"""
df_result['participant_number_clean'] = df_result['participant_number'].str.split('_').str[0]
df['numAppleGame'] = df['numAppleGame'].apply(lambda x: str(x).zfill(3))
df_result = pd.merge(df_result, df[['numAppleGame', 'sexe']], 
                     left_on='participant_number_clean', right_on='numAppleGame', 
                     how='left')


df_result.drop(columns=['participant_number_clean'], inplace=True)
df_result.drop(columns=['numAppleGame'], inplace=True)

"""
Rename pour facilité
"""
df_result = df_result.rename(columns={'sexe': 'Gender'})
df_result = df_result.rename(columns={'AnneeEtude': 'Education'})
df_result = df_result.rename(columns={'Nbre_heure_jeux_videos': 'VideoGame'})


"""
Ajout groupe d'âge
"""
bins = [0, 51, 61, 71, 100] 
labels = ['<=50', '51-60', '61-70', '>70']
df_result['AgeGroup'] = pd.cut(df_result['age'], bins=bins, labels=labels, right=False)

"""
Séparation de participant_number en block_number et Participant
"""
df_result['block_number'] = df_result['participant_number'].str.split('_').str[1].astype(int)
df_result['Participant'] = df_result['participant_number'].str.split('_').str[0]
df_result.drop(columns=['participant_number'], inplace=True)


"""
Enlever les blocs contenant moins de 6 essais
"""
print("Filtrage: suppression des blocs < 6 essais...")
trial_counts = df_result.groupby(["Participant", "block_number"])["trial_number"].nunique()

valid_blocks = trial_counts[trial_counts >= 6].index

df_result = df_result[df_result.set_index(["Participant", "block_number"]).index.isin(valid_blocks)].reset_index(drop=True)

trial_counts_filtered = df_result.groupby(["Participant", "block_number"])["trial_number"].nunique()
print(f"-> Nombre d'essais par bloc (après filtrage) Moyenne: {trial_counts_filtered.mean():.2f}, Min: {trial_counts_filtered.min()}, Max: {trial_counts_filtered.max()}")
"""
Retirer le bloc d'entrainement (bloc 6 pour la version 82, bloc 1 pour les autres versions)
"""
print("Filtrage: retrait des blocs d'entraînement...")
df_result = df_result[~((df_result['block_number'].isin([6])) & (df_result['VersionAppleGame'] == 82))]
df_result = df_result[~((df_result['block_number'].isin([1])) & (df_result['VersionAppleGame'] != 82))]
print(f"-> Différentes valeurs de block_number pour la version 82 {df_result[(df_result['VersionAppleGame'] == 82)]['block_number'].unique()}")
print(f"-> Différentes valeurs de block_number pour les autres versions {df_result[(df_result['VersionAppleGame'] != 82)]['block_number'].unique()}")

"""
Retirer les participants ayant moins de 2 blocs sans compter celui d'entrainement
"""
print("Filtrage: retrait des participants avec < 2 blocs...")
participant_block_counts = df_result.groupby("Participant")["block_number"].nunique()

participants_few_blocks = participant_block_counts[participant_block_counts < 2].index

df_result = df_result[df_result["Participant"].isin(participant_block_counts[participant_block_counts >= 2].index)]


"""
Ajout de l'APOE: 1 carriers, 0 non carriers
"""
print("Ajout des valeurs APOE ... ")
def apoe_binary(apoe_value):
    if apoe_value in [34, 44]:
        return 1  
    elif apoe_value in [23, 22, 33]:
        return 0  
    else:
        return None 

df['APOE'] = df['APOE'].apply(apoe_binary)


df = df[df['APOE'].notnull()]


df_result = pd.merge(df_result, df[['numAppleGame', 'APOE']], 
                     left_on='Participant', right_on='numAppleGame', 
                     how='left')

df_result = df_result.dropna(subset=['APOE'])

df_result.drop(columns=[ 'numAppleGame'], inplace=True)

print(f"-> Différentes valeurs de APOE {df_result['APOE'].unique()}")

"""
Ajout MMSE value et retirer les participants ayant un MMSE<25
"""
print("Ajout des valeurs MMSE et filtrage MMSE < 25 ...")
def mmse(mmse_value, age):
    if pd.isna(mmse_value) and age < 40:
        return 30  # On considère les jeunes comme non déments par défaut
    elif (pd.isna(mmse_value) or mmse_value < 25.0):
        return None  # Exclure les participants avec MMSE < 25 ou NaN
    else:
        return mmse_value  # Garder normalement

df['MMSE'] = df.apply(lambda row: mmse(row['MMSESur30'], row['age']), axis=1)


df_result= pd.merge(df_result, df[['numAppleGame', 'MMSE']], 
                     left_on='Participant', right_on='numAppleGame', 
                     how='left')

df_result = df_result.dropna(subset=['MMSE'])
df_result.drop(columns=['numAppleGame'], inplace=True)
print(f"-> Différentes valeurs de MMSE après ajout et filtrage {df_result['MMSE'].unique()}")


# Sauvegarder dans un fichier csv
df_result.to_csv('errors.csv', index=False)