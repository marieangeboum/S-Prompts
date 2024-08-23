import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm.models import create_model
import timm

# Charger le modèle ViT pré-entraîné
model_name = 'vit_base_patch16_224'  # Choisir le modèle ViT que vous souhaitez évaluer
model = timm.create_model(model_name, pretrained=True)
model.eval()

# Prétraiter l'image d'exemple
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])

# Charger une image d'exemple
image_path = '/run/user/108646/gvfs/sftp:host=flexo/scratcht/Dep_Traj/D004_2021/UU/IMG_000331.png'  # Remplacez par le chemin de votre image
image_ = Image.open(image_path)

image = transform(image_).unsqueeze(0)  # Ajouter une dimension de lot (batch dimension)

# Effectuer une inférence "zero-shot"
with torch.no_grad():
    output = model(image)

# Interpréter les résultats
probabilities = F.softmax(output[0], dim=0)
label_indices = torch.argsort(output[0], descending=True)

# Afficher les 5 premières classes prédites
top_k = 5
for i in range(top_k):
    label_idx = label_indices[i].item()
    label_prob = probabilities[label_idx].item()
    print(f"Classe {label_idx}: Probabilité {label_prob:.4f}")

# Note : Vous devrez également avoir les étiquettes de classe pour interpréter les résultats en fonction de votre ensemble de données.
