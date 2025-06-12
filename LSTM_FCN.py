import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import scipy
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("errors.csv")

df['MMSE2'] = np.where(df['MMSE'] >= 27, 1, 0)


class TimeSeriesDataset(Dataset):
    #dataset avec des séquences temporelles et leurs labels associés.
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """
    Fonction utilisée par le DataLoader  qui effectue le padding.
    Args:
        batch (list of tuples): liste de paires (séquence, label), comme renvoyée par TimeSeriesDataset.
    Returns:
        sequences_padded : séquences remplies avec du padding.
        labels : labels correspondants à chaque séquence .
        lengths : longueurs originales des séquences avant le padding.
    """
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    sequences_padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)

class SqueezeExciteBlock(nn.Module):
    """
    Bloc 'Squeeze and Excite' 
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        se = F.adaptive_avg_pool1d(x, 1).view(batch_size, channels)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(batch_size, channels, 1)
        return x * se.expand_as(x)

class FCN_LSTM(nn.Module):
    """
    Architecture du réseau avec 2 branches:
    - la branche FCN
    - la branche LSTM 
    """
    def __init__(self, input_channels, num_classes):
        super(FCN_LSTM, self).__init__()

        # FCN branch
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.se1 = SqueezeExciteBlock(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.se2 = SqueezeExciteBlock(64)

        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)

        # LSTM branch
        self.lstm = nn.LSTM(input_channels, 32, batch_first=True)
        self.dropout = nn.Dropout(p=0.4)

        # Final classification
        self.fc = nn.Linear(32 + 32, num_classes)

    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, input_channels)
        x_fcn = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_len)

        x_fcn = F.relu(self.bn1(self.conv1(x_fcn)))
        x_fcn = self.se1(x_fcn)

        x_fcn = F.relu(self.bn2(self.conv2(x_fcn)))
        x_fcn = self.se2(x_fcn)

        x_fcn = F.relu(self.bn3(self.conv3(x_fcn)))
        x_fcn = F.adaptive_avg_pool1d(x_fcn, 1).squeeze(-1)  # (batch_size, 128)

        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        x_lstm = self.dropout(hn[-1])
        #x_lstm = hn[-1]  # last time step (batch_size, 128)

        x_combined = torch.cat([x_fcn, x_lstm], dim=1)
        out = self.fc(x_combined)
        return out

def load_mat_data(filepath):
    # Charge un fichier .mat MATLAB
    data = scipy.io.loadmat(filepath)
    keys = [key for key in data.keys() if not key.startswith('__')]  
    return {key: np.array(data[key]) for key in keys}

def load_labels(label_file):
    """
    Charge les labels depuis un fichier CSV et transforme la variable MMSE en variable binaire.

    Args:
        label_file (str): chemin vers le fichier CSV contenant les colonnes 'Participant' et 'MMSE'.

    Returns:
        dict: dictionnaire {Participant_ID: MMSE_binaire}
              où MMSE_binaire vaut 1 si MMSE >= 27, sinon 0.
    """
    df = pd.read_csv(label_file)
    df['MMSE2'] = np.where(df['MMSE'] >= 27, 1, 0)
    labels_dict = dict(zip(df['Participant'], df['MMSE2']))
    return labels_dict

def load_version(version_file):
    """
    Charge les versions de l'Apple Game depuis un fichier CSV.

    Args:
        label_file (str): chemin vers le fichier CSV contenant les colonnes 'Participant' et 'VersionAppleGame'.

    Returns:
        dict: dictionnaire {Participant_ID: VersionAppleGame}
    """
    df = pd.read_csv(version_file)
    versions_dict = dict(zip(df['Participant'], df['VersionAppleGame']))
    return versions_dict

def load_tree(tree_file):
    """
    Charge le nombre d'arbre depuis un fichier CSV.

    Args:
        label_file (str): chemin vers le fichier CSV contenant les colonnes 'Participant' et 'tree_count'.

    Returns:
        dict: dictionnaire {Participant_ID: tree_count}
    """
    df = pd.read_csv(tree_file)
    tree_dict = {
        (row['Participant'], row['block_number'], row['trial_number']): row['tree_count']
        for _, row in df.iterrows()
    }
    return tree_dict

def augment_trajectory(sequence, gaussien_std=0.1, scale_range=(0.9, 1.1)):
    """
    Applique une augmentation de données sur une séquence de trajectoire 2D.

    Augmentations utilisées :
        - Bruit gaussien (jitter) sur les coordonnées x et y
        - Mise à l'échelle (scaling) aléatoire uniforme

    Args:
        sequence (Tensor): séquence d'entrée
        jitter_std (float): écart-type du bruit gaussien ajouté à chaque point
        scale_range (tuple): intervalle (min, max) pour le facteur d'échelle aléatoire

    Returns:
        Tensor: nouvelle séquence augmentée
    """
    augmented = sequence.clone()

    # Bruit gaussien
    augmented[:, 0] += torch.normal(0, gaussien_std, size=augmented[:, 0].shape)
    augmented[:, 1] += torch.normal(0, gaussien_std, size=augmented[:, 1].shape)

    # Scaling
    scale = np.random.uniform(*scale_range)
    augmented[:, 0] *= scale
    augmented[:, 1] *= scale

    return augmented

def load_dataset(folder_path, label_file):
    """
    Création des séquences x avec le label correspondant y.
    Args:
        folder_path: chemin du dossier contenant les logfiles
        label_file: chemin du fichier excel contenant le label, la version de l'Apple Game et le nombre d'arbre dans un essai
    Returns:
        sequences: les sequences (correspondant chacune à un essai) avec x, y (la position du participant) correct_x, correct_y, trial (one hot encoded) et phase (one hot encoded)
        labels: labels correspondants (MMSE >=27 ou MMSE <27)
    """
    sequences = []
    labels = []
    labels_dict = load_labels(label_file)
    versions_dict = load_version(label_file)
    tree_count_dict = load_tree(label_file)
    expected_phases_1 = {1, 3,4} 
    expected_phases = {1, 2, 3,4} 
    phase_column = 9
    all_trials = np.array([[i] for i in range(1, 9)])  # suppose que goals vont de 1 à 8
    all_phases = np.array([[i] for i in range(1, 5)])  # suppose que trials vont de 1 à 4 4+8+2+2
    encoder_trials = OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(all_trials)
    encoder_phases = OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(all_phases)
    for root, _, files in os.walk(folder_path):
        indices_to_exclude = [0, 3,4, 5,6,7,8,9]
        for file in files:
            if file.endswith(".mat"):
                participant_id = int(file.split("_")[0])  # Extract participant ID

                block_number = int(file.split('_')[1])
                training_block = False
                if participant_id in versions_dict.keys():
                    if(block_number == 1 and versions_dict[participant_id]!=82):
                        training_block = True
                    if(block_number == 6 and versions_dict[participant_id]==82):
                        training_block = True   
                if participant_id in labels_dict.keys() and not training_block:
                    data = load_mat_data(os.path.join(root, file))
                    time_series = data['data']
                    indices_to_exclude += list(np.arange(12, time_series.shape[1]))
                    time_series = torch.tensor(time_series, dtype=torch.float32)
                    filtered_data = time_series[(time_series[:, 10] != -999.0) & (time_series[:, 11] != -999.0)]
                    
                    trials = filtered_data[:, 8].numpy().reshape(-1, 1)  
                    phases = filtered_data[:, 9].numpy().reshape(-1, 1)  
                    
                    # Appliquer l'encodage One-Hot sur les deux colonnes
                    trials_encoded = encoder_trials.transform(trials)
                    phases_encoded = encoder_phases.transform(phases)
                    
                    
                    # Créer un tensor pour les encodages One-Hot
                    trials_tensor = torch.tensor(trials_encoded, dtype=torch.float32)
                    
                    phases_tensor = torch.tensor(phases_encoded, dtype=torch.float32)
                    # Ajouter les variables One-Hot encodées aux données filtrées
                    encoded_data = torch.cat((filtered_data, trials_tensor,phases_tensor), dim=1)
                    unique_trials = torch.unique(encoded_data[:, 8])  # Identifier les trials
                    for trial in unique_trials:
                        #si on veut prendre que les essais avec un seul arbre
                        #if( tree_count_dict[(participant_id, block_number, int(trial))]==1): 
                        trial_data = encoded_data[encoded_data[:, 8] == trial]
                        trial_phases = set(trial_data[:, phase_column].int().tolist())
                        if trial_phases == expected_phases or trial_phases == expected_phases_1:
                            trial_data = torch.cat([trial_data[:, i].unsqueeze(1) for i in range(trial_data.shape[1]) if i not in indices_to_exclude], dim=1)
                            if(trial_data.shape[0]>0):
                                prob_augment = 0.0 # probalibilité de data augmentation (à modifier comme souhaité)
                                sequences.append(trial_data)
                                labels.append(int(labels_dict[participant_id]))
                                if np.random.rand() < prob_augment:
                                    # Appliquer l'augmentation
                                    augmented = augment_trajectory(trial_data)
                                    sequences.append(augmented)
                                    labels.append(int(labels_dict[participant_id]))
                                
    labels = torch.tensor(labels, dtype=torch.long)

    return sequences, labels

def subsample_sequence(seq, target_length=300):
    L = len(seq)

    if L <= target_length:
        return seq

    indices = torch.linspace(0, L-1, target_length).long()

    if isinstance(seq, torch.Tensor):
        indices = indices.to(seq.device) 
        return torch.index_select(seq, dim=0, index=indices)
    else:
        return [seq[i] for i in indices.tolist()]
    

folder_path = "data_TFE_24_25/logfiles" #chemin vers le dossier contenant les logfiles
label_file = "errors_with_age.csv"  # chemin du fichier excel contenant les labels
x, y = load_dataset(folder_path, label_file)

#Analyse des longueurs des séquences
longueurs = [len(sequence) for sequence in x]
print(f"Nombre de séquences : {len(longueurs)}")
print(f"Longueur minimale : {min(longueurs)}")
print(f"Longueur maximale : {max(longueurs)}")
print(f"Longueur moyenne : {sum(longueurs)/len(longueurs):.2f}")

#subsampling des séquences plus longues
x_new = []
for xi in x:
    x_new.append(subsample_sequence(xi))

#Splitting du dataset 80% training set, 20% validation set
x_train, x_val, y_train, y_val = train_test_split(x_new, y, test_size=0.2, random_state=42, stratify=y)

# Standardisation des données
all_train_points = np.concatenate(x_train, axis=0) 

scaler = StandardScaler()
scaler.fit(all_train_points)

x_train_scaled = [scaler.transform(traj) for traj in x_train]
x_val_scaled   = [scaler.transform(traj) for traj in x_val]

#Création des DataLoader avec un batch size =32 + padding des séquences plus courtes grâce à collate_fn
train_dataset = TimeSeriesDataset(x_train_scaled, y_train)
val_dataset = TimeSeriesDataset(x_val_scaled, y_val)
batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Décommenter les lignes qui suivent pour obtenir la matrice de confusion d'un modèle sauvegardé
#model.load_state_dict(torch.load("model/best_LSTM_FCN7.pth"))
#model.eval()


#all_preds = []
#all_true = []

#with torch.no_grad():
#    for x_val_batch, y_val, lengths_val in val_loader:
#        output_val = model(x_val_batch, lengths_val)
#        preds = torch.argmax(output_val, dim=1)
#        all_preds.extend(preds.cpu().numpy())
#        all_true.extend(y_val.cpu().numpy())

#pred_counter = Counter(all_preds)
#true_counter = Counter(all_true)

#print("🔍 Distribution des prédictions :", pred_counter)
#print("✅ Distribution réelle des labels :", true_counter)
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#cm = confusion_matrix(all_true, all_preds)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#disp.plot(cmap="Blues")
#plt.title("Confusion matrix on the validation set")
#plt.show()


#Création du modèle
model = FCN_LSTM(input_channels=16, num_classes=2)

#Calcul des poids pour la fonction de perte
y_train_list = y_train.tolist() 
class_counts = Counter(y_train_list)
total = sum(class_counts.values())
weights = [total / class_counts[0], total / class_counts[1]]
class_weights = torch.tensor(weights, dtype=torch.float32)

#Fonction de perte avec les poids
criterion = nn.CrossEntropyLoss(weight=class_weights)

#Optimizer
optimizer = torch.optim.Adam(list(model.parameters()) , lr=1e-4, weight_decay=1e-3)
#Sheduler
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode="triangular2", base_lr=5e-6, max_lr=1e-4)

num_epochs = 100 #Nombre d'epoch


train_loss_history = []
val_loss_history = []

# Training loop
print("🚀 Training started...")
best_val_loss = float('inf')
best_model_path = "model/best_LSTM_FCN8.pth"

for epoch in range(num_epochs):
    #training du modèle et update des poids
    epoch_loss = 0.0
    model.train()
    losses = []
    for x, y, lengths in train_loader: 

        optimizer.zero_grad()
        output = model(x, lengths)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()

    #évaluation sur le validation set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val_batch, y_val, lengths_val in val_loader:
            output_val= model(x_val_batch, lengths_val)  
            val_loss = criterion(output_val, y_val)
            val_losses.append(val_loss.item())
    
    #Calcul des pertes et sauvegarde du modèle si amélioration de la perte sur le validation set
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_loss = sum(losses) / len(losses)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Nouveau meilleur modèle sauvegardé avec val_loss={avg_val_loss:.4f}")
    
    #création du graphe des pertes en fonction des epochs
    train_loss_history.append(avg_loss)
    val_loss_history.append(avg_val_loss)
    print(f"Epoch {epoch:2d}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Évolution des pertes pendant l'entraînement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plot/loss_FCN_LSTM_plot8.png")  # ← Sauvegarde ici
    plt.close()

    # Sauvegarde des listes de pertes
    torch.save({
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }, "Plot/loss_history_FCN_LSTM8.pt")

    print("📁 Fichiers 'loss_FCN_LSTM_plot8.png' et 'loss_history_FCN_LSTM8.pt' enregistrés.")
print("✅ Training completed!")
