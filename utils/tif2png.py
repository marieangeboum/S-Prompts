from PIL import Image
import os

#/scratcht/FLAIR_1_Custom_912021/Z1_UA/train/0_no_building
# Dossier contenant les fichiers TIFF à convertir
dossier_entree = "/scratcht/FLAIR_1/train/D091_2021/*/img/IMG_*.tif"
# Dossier de sortie où seront enregistrées les images PNG converties
dossier_sortie = "/scratcht/FLAIR_1_Projet/D091_2021/img/"

# Assurez-vous que le dossier de sortie existe, sinon créez-le
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# Parcourez tous les fichiers dans le dossier d'entrée
for filename in os.listdir(dossier_entree):
    if filename.endswith(".tif"):
        # Ouvrez le fichier TIFF
        image = Image.open(os.path.join(dossier_entree, filename))

        # Remplacez l'extension du fichier par .png
        nom_fichier_png = os.path.splitext(filename)[0] + ".png"

        # Enregistrez l'image au format PNG dans le dossier de sortie
        image.save(os.path.join(dossier_sortie, nom_fichier_png), "PNG")

        # Fermez le fichier TIFF
        image.close()

        # Supprimez le fichier TIFF
        os.remove(os.path.join(dossier_entree, filename))