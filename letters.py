# -*- coding: utf-8 -*-

#----------------------------------------------------------------------
#Auteur : Théo Guidoux, stagiaire 5 sem 1A
#Python : 2.7
#Description : Ce programme est destiné à donner une première approche du Machine Learning avec SciKit Learn
#Ce programme est un programme de test qui permet de comprendre le fonctionnement du Machine Learning, apprentissage supervisé avec plusieurs algorithmes d'apprentissage
#C'est un programme qui permet de reconnaitre des caractères en majuscules.
#Chaque caractère est caractérisé par 16 paramètres, nombre de bar, position, longueur...
#Chaque caractère correspond à un code Ex : A => 789
#L'algorithme sera capable de sortir un nombre qui devra ensuite être convertie en lettre grâce à convert_target_to_str
#Les données d'entrée ont été téléchargées à partir de http://www.cs.toronto.edu/~delve/data/datasets.html et sont disponibles dans le fichier data.csv
#
#En faisant varier nb_entree (nombre d'entrée prises pour apprendre les lettres) et l'algorithme d'apprentissage (classifier)
#on observe que la précision (score) varie également.
#Plus le nombre d'entrée est important, plus la précision est optimale



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

def convert_target_to_int(argument):
	#ENTREE => argument (char)
	#SORTIE => nombre correspondant à argument (int)
	#Fonction qui fait correspondre une lettre majuscule à un nombre selon la doc des données

    switcher = {
    'A': 789, 	   'B': 766,      'C': 736,      'D': 805, 	 'E': 768, 	   'F': 775,      'G': 773,
 	'H': 734, 	   'I': 755,      'J': 747,      'K': 739, 	 'L': 761, 	   'M': 792,      'N': 783,
 	'O': 753, 	   'P': 803,      'Q': 783,      'R': 758, 	 'S': 748, 	   'T': 796,      'U': 813,
 	'V': 764, 	   'W': 752,      'X': 787,      'Y': 786, 	 'Z': 734
 	}

    return switcher.get(argument, "nothing")

def convert_target_to_str(argument):
	#ENTREE => argument (int)
	#SORTIE => lettre correspondant à argument (char)
	#Fonction qui fait correspondre une lettre majuscule à un nombre selon la doc des données

    switcher = {
    789: 'A', 	   766: 'B',      736: 'C',      805: 'D', 	 768: 'E', 	   775: 'F',      773: 'G',
 	734: 'H', 	   755: 'I',      747: 'J',      739: 'K', 	 761: 'L', 	   792: 'M',      783: 'N',
 	753: 'O', 	   803: 'P',      783: 'Q',      758: 'R', 	 748: 'S', 	   796: 'T',      813: 'U',
 	764: 'V', 	   752: 'W',      787: 'X',      786: 'Y', 	 734: 'Z'
 	}

    return switcher.get(argument, "nothing")

print("Lecture des donnees ...")
read_data = pd.read_csv('data.csv')

#liste des sorties qui correspondent aux entrées
target = read_data['lettr']

#liste qui correspond aux 16 paramètres de chaque lettre
data = [read_data['width'], read_data['high'], read_data['onpix'], read_data['X-bar'], read_data['Y-bar'], read_data['x2bar'], read_data['y2-bar'], read_data['xybar'], read_data['x2ybr'], read_data['xy2br'], read_data['x-ege'], read_data['xegvy'], read_data['y-ege'], read_data['yegvx']]

print("Preparation des donnees...")
structured_target = []
structured_data = []

nb_entree = len(target)#Interessant de modifier nb_entree (< 20000)

#restructuration des sorties en nombres
for i in  range(0, nb_entree):
	structured_target.append(convert_target_to_int(target[i]))

#restructuration des entrées de 16*20000 à 20000*16
for i in  range(0, len(structured_target)):
	structured_data.append([read_data['x-box'][i], read_data['y-box'][i], read_data['width'][i], read_data['high'][i], read_data['onpix'][i], read_data['X-bar'][i], read_data['Y-bar'][i], read_data['x2bar'][i], read_data['y2-bar'][i], read_data['xybar'][i], read_data['x2ybr'][i], read_data['xy2br'][i], read_data['x-ege'][i], read_data['xegvy'][i], read_data['y-ege'][i], read_data['yegvx'][i]])

#Interessant de modifier l'algorithme d'apprentissage
#classifier = svm.SVC()
#classifier = linear_model.LinearModel()
classifier = neighbors.KNeighborsClassifier()
#classifier = tree.DecisionTreeClassifier()

print("Apprentissage...")
classifier.fit(structured_data, structured_target)

#I
#donnee_test = [5 ,12 ,3 ,7 ,2 ,10 ,5 ,5 ,4 ,13 ,3 ,9 ,2 ,8 ,4 ,10]

#T num : 1
#donnee_test = [2, 8, 3, 5, 1, 8, 13, 0, 6, 6, 10, 8, 0, 8, 0, 8]

#donnee_test = [2, 8, 3, 5, 1, 8, 13, 0, 6, 6, 10, 8, 0, 8, 0, 7]

#T num : 5923
#donnee_test = [6, 7, 6, 5, 3, 4, 13, 4, 6, 12, 9, 4, 1, 11, 2, 4]

#T num : 5923 modifié legèrement (pour observer le comportement lorsque qu'une valeur jamais vue est en entrée)
donnee_test = [6, 7, 6, 5, 3, 4, 13, 4, 6, 12, 9, 4, 1, 11, 2, 3]

print("Prediction de ",donnee_test)
result = classifier.predict([donnee_test])

print("Resultat :"+str(result[0]))
print(str(result[0]) + " => " + convert_target_to_str(result[0]))

print("Precision : " + str(classifier.score(structured_data, structured_target)))

#En mettant nb_entree à 100 et en testant le 'T' num 5923, on observe que l'algorithme se trompe.
#Quand nb_entree augmente, la sortie se trompe jusqu'à un certain seuil
