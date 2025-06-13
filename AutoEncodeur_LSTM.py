import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import os
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

RANDOM_SEED= 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class TimeSeriesDataset(Dataset):
    #dataset avec des séquences temporelle
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    """
    Fonction utilisée par le DataLoader  qui effectue le padding.
    Args:
        batch (list of tuples): liste de sequences, comme renvoyée par TimeSeriesDataset.
    Returns:
        sequences_padded : séquences remplies avec du padding.
        lengths : longueurs originales des séquences avant le padding.
    """
    sequences = batch
    lengths = [len(seq) for seq in sequences]
    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    sequences_padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.tensor(lengths)


class Encoder(nn.Module):
    
    def __init__(self, n_features, embedding_dim=64):
        """
        L'encodeur prend une séquence en entrée et la compresse en un vecteur latent.

        Args:
            n_features (int): nombre de variables (dimensions) à chaque pas de temps
            embedding_dim (int): dimension du vecteur latent
        """
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )
    def forward(self, x,lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn1(packed)
        packed_output, (hidden_n, _) = self.rnn2(packed_output)
        return hidden_n[-1]

class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        """
        Le décodeur reconstruit la séquence d'origine à partir du vecteur latent.

        Args:
            seq_len (int): longueur de séquence à générer
            input_dim (int): dimension du vecteur latent
            n_features (int): nombre de dimensions de sortie à chaque pas de temps
        """
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x,lengths):
        batch_size = x.size(0)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        #x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)
    
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        """
        Autoencodeur séquentiel basé sur LSTM avec encodeur et décodeur.

        Args:
            seq_len (int): longueur des séquences
            n_features (int): nombre de variables par pas de temps
            embedding_dim (int): taille de l’espace latent
        """
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
    def forward(self, x,length):
        x = self.encoder(x,length)
        x = self.decoder(x,length)
        return x    


def load_mat_data(filepath):
    # Charge un fichier .mat MATLAB
    data = scipy.io.loadmat(filepath)
    keys = [key for key in data.keys() if not key.startswith('__')]  
    return {key: np.array(data[key]) for key in keys}

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

def augment_trajectory(sequence, gaussien_std=0.1, scale_range=(0.9, 1.1)):
    """
    Applique une augmentation de données sur une séquence de trajectoire 2D.

    Augmentations utilisées :
        - Bruit gaussien sur les coordonnées x et y
        - Mise à l'échelle (scaling) aléatoire uniforme

    Args:
        sequence (Tensor): séquence d'entrée
        gaussien_std (float): écart-type du bruit gaussien ajouté à chaque point
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
    Création des séquences x.
    Args:
        folder_path: chemin du dossier contenant les logfiles
        label_file: chemin du fichier excel contenant la version de l'Apple Game
    Returns:
        sequences: les sequences (correspondant chacune à un essai) avec x, y (la position du participant)
    """
    sequences = []
    versions_dict = load_version(label_file)

    expected_phases_1 = {1, 3,4} 
    expected_phases = {1, 2, 3,4} 
    phase_column = 9
    
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
                if participant_id in versions_dict.keys() and not training_block:
                    data = load_mat_data(os.path.join(root, file))
                    time_series = data['data']
                    indices_to_exclude += list(np.arange(12, time_series.shape[1]))
                    time_series = torch.tensor(time_series, dtype=torch.float32)
                    filtered_data = time_series[(time_series[:, 10] != -999.0) & (time_series[:, 11] != -999.0)]

                    encoded_data = filtered_data
                    unique_trials = torch.unique(encoded_data[:, 8])  # Identifier les trials
                    for trial in unique_trials:
                        trial_data = encoded_data[encoded_data[:, 8] == trial]
                        trial_phases = set(trial_data[:, phase_column].int().tolist())
                        if trial_phases == expected_phases or trial_phases == expected_phases_1:
                            trial_data = trial_data[:, [1, 2]] # on recupère seulement le x, y
                            #trial_data = torch.cat([trial_data[:, i].unsqueeze(1) for i in range(trial_data.shape[1]) if i not in indices_to_exclude], dim=1)
                            if(trial_data.shape[0]>0):
                                prob_augment = 0.3  # probalibilité de data augmentation (à modifier comme souhaité)
                                sequences.append(trial_data)
                                if np.random.rand() < prob_augment:
                                    # Appliquer l'augmentation
                                    augmented = augment_trajectory(trial_data)
                                    sequences.append(augmented)
                                                             
    return sequences

def masked_mse_loss(preds, targets, lengths):
    """
    Calcule la perte MSE uniquement sur les parties valides des séquences (en ignorant le padding).

    Args:
        preds (Tensor): prédictions du modèle
        targets (Tensor): valeurs cibles, même forme que preds
        lengths (Tensor): longueurs réelles de chaque séquence

    Returns:
        loss (float): MSE moyenne sur les valeurs non masquées (non paddées)
    """
    _, max_len, input_dim = preds.size()
    
    # Créer un masque booléen (batch_size, max_len)
    mask = torch.arange(max_len, device=preds.device)[None, :] < lengths[:, None]
    
    # Ajouter la dimension de features pour faire (B, T, F)
    mask = mask.unsqueeze(-1).expand(-1, -1, input_dim) 

    loss = ((preds - targets) ** 2)
    loss = loss * mask
    return loss.sum() / mask.sum()

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
label_file = "errors.csv"  # chemin du fichier excel contenant la version de l'Apple Game
x= load_dataset(folder_path, label_file)

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
x_train, x_val = train_test_split(x_new, test_size=0.2, random_state=42)

# Standardisation des données
all_train_points = np.concatenate(x_train, axis=0) 

scaler = StandardScaler()
scaler.fit(all_train_points)

x_train_scaled = [scaler.transform(traj) for traj in x_train]
x_val_scaled   = [scaler.transform(traj) for traj in x_val]

#Création des DataLoader avec un batch size =32 + padding des séquences plus courtes grâce à collate_fn
train_dataset = TimeSeriesDataset(x_train_scaled)
val_dataset = TimeSeriesDataset(x_val_scaled)
batch_size = 32 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


#Création du modèle
model = RecurrentAutoencoder(300, 2, 128)
#Optimizer
optimizer = optim.Adam(list(model.parameters()) , lr=1e-4, weight_decay=1e-4)
#Sheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience =2)

train_loss_history = []
val_loss_history = []
# Training loop
print("Training started...")
best_val_loss = float('inf')
best_model_path = "model/best_model_6.pth"
for epoch in range(50):
    epoch_loss = 0.0
    model.train()
    losses = []

    for x, lengths in train_loader: 
        x = x.float()
        optimizer.zero_grad()

        reconstructed = model(x, lengths)
        
        loss = masked_mse_loss(reconstructed, x, lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    #évaluation sur le validation set
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_val_batch, lengths_val in val_loader:
            x_val_batch = x_val_batch.float()
            reconstructed_val= model(x_val_batch, lengths_val) 
            val_loss = masked_mse_loss(reconstructed_val, x_val_batch, lengths_val)
            val_losses.append(val_loss.item())
    
     #Calcul des pertes et sauvegarde du modèle si amélioration de la perte sur le validation set
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Nouveau meilleur modèle sauvegardé avec val_loss={avg_val_loss:.4f}")

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
    plt.savefig("Plot/loss_plot.png")
    plt.close()

    #Sauvegarde des listes de pertes
    torch.save({
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }, "Plot/loss_history5.pt")

    print("Fichiers 'loss_plot.png' et 'loss_history.pt' enregistrés.")
torch.save(model.state_dict(), "model/best_model_5_final.pth")
print("Training completed!")



