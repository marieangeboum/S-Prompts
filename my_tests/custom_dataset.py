import os
import random
import numpy as np
import shutil
import glob

target_domain_expe = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1/train/D091_2021/"

root_flair_custom_mab = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1_Custom_912021/"
domaines_custom = os.listdir(target_domain_expe)

# Boucle pour créer chaque sous-dossier
for sous_dossier in domaines_custom :
    chemin_sous_dossier = os.path.join(root_flair_custom_mab, sous_dossier)
    os.makedirs(chemin_sous_dossier)
    # Créez les sous-dossiers "train" et "val" dans chaque sous-dossier
    chemin_train = os.path.join(chemin_sous_dossier, "train")
    chemin_val = os.path.join(chemin_sous_dossier, "val")
    os.makedirs(chemin_train)
    os.makedirs(chemin_val)
    chemin_train_building = os.path.join(chemin_train, "1_building")
    chemin_train_no_building = os.path.join(chemin_train, "0_no_building")
    chemin_val_building = os.path.join(chemin_val, "1_building")
    chemin_val_no_building = os.path.join(chemin_val, "0_no_building")
    os.makedirs(chemin_train_building)
    os.makedirs(chemin_val_building)
    os.makedirs(chemin_train_no_building)
    os.makedirs(chemin_val_no_building)

building_data = {key: value for key, value in metadata.items() if value['domain'] == 'D091_2021' and  1 in value['labels'] }
no_building_data = {key: value for key, value in metadata.items() if value['domain'] == 'D091_2021' and  1 not in value['labels'] }

building_data_imgs = list(building_data.keys())
no_building_data_imgs = list(no_building_data.keys())
for zone in domaines_custom : 
    destination_path_train_building = f'/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1_Custom_912021/{zone}/train/1_building/'
    destination_path_train_no_building = f'/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1_Custom_912021/{zone}/train/0_no_building/'
    destination_path_test_building = f'/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1_Custom_912021/{zone}/val/1_building/'
    destination_path_test_no_building = f'/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1_Custom_912021/{zone}/val/0_no_building/'
    
    imgs_list = glob.glob(f'/run/user/108646/gvfs/sftp:host=flexo/scratcht/FLAIR_1/train/D091_2021/{zone}/img/IMG_*.tif')
    # Calculer le nombre d'éléments à sélectionner (80 % de la longueur de la liste)
    pourcentage = 0.8
    nombre_a_selectionner = int(len(imgs_list) * pourcentage)
    train = random.sample(imgs_list, nombre_a_selectionner)
    test = [element for element in imgs_list if element not in train]
    for img_path in train :
        if img_path.split('/')[-1].strip('.tif')  in building_data_imgs :
            # Copy the file from the source to the destination
            shutil.copy(img_path, destination_path_train_building)
        else : 
            shutil.copy(img_path, destination_path_train_no_building)
    for img_path in test : 
        if img_path.split('/')[-1].strip('.tif')  in building_data_imgs :
            # Copy the file from the source to the destination
            shutil.copy(img_path, destination_path_test_building)
        else : 
            shutil.copy(img_path, destination_path_test_no_building)