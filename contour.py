import cv2
import numpy as np
from math import pi
import json


def detection_dent(segmentation, imageDeBase):
    dent = imageDeBase.copy();

    contours, hierarchy = cv2.findContours(segmentation, 1, cv2.CHAIN_APPROX_TC89_KCOS)
    # methode d'approximation (dernier argument : maybe CHAIN_APPROX_TC89_L1 resultats proches )
    #on considere que le contours de la feuille est le contour le plus long :

    contourUtile = max(contours, key=len)
    # cv2.drawContours(imageDeBase, contourUtile, -1, (0,0,255), 2)
    # cv2.imshow("contour utilise", imageDeBase)

    list_contour = [] #pour stocker les points qui sont consideres comme des dents
    #on parcourt tous les points du contour et on regarde s'ils sont a peu pres alignes
    #si l'angle entre les points est trop important on considere qu'il y a une dent
    #On regarde si les points sont alignes en utilisant l'inegalite triangulaire (aligne si AB+BC = AC)
    for i in range(len(contourUtile) - 2):
        a = contourUtile[i]
        b = contourUtile[i + 1]
        c = contourUtile[i + 2]
        AB = np.linalg.norm(b - a)
        BC = np.linalg.norm(c - b)
        AC = np.linalg.norm(c - a)
        if ((AC / (AB + BC)) < 0.926): #valeur determiner par l'experience : seuil a partir duquel on considere que les points ne sont pas alignes
            list_contour.append(b)
    #Affichage des dents
    # cv2.drawContours(dent, list_contour, -1, (0, 0, 255), 3)
    # cv2.imshow("dents de la feuille", dent)

    #nombre de dents
    nombreDent = len(list_contour)
    #On fait un seuil a partir duquel on considere qu'il y a des dents
    presenceDent = False
    if nombreDent > 20 :
        presenceDent=True
    return presenceDent


def feuille_convexe(segmentation, imageDeBase):

    contours, hierarchy = cv2.findContours(segmentation, 1, cv2.CHAIN_APPROX_TC89_KCOS)
    # on considere que le contours de la feuille est le contour le plus long :
    contourUtile = max(contours, key=len)

    #On recupere le contour convexe de notre feuille
    hull = cv2.convexHull(contourUtile, returnPoints=False)
    #On recupere les defauts de convexite de notre contour
    defects = cv2.convexityDefects(contourUtile, hull)
    defautDetecte=0;
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contourUtile[s][0]) #point du contours convexe avant
        end = tuple(contourUtile[e][0])   #point du contours convexe d'apres
        far = tuple(contourUtile[f][0])   #defaut de convexite entre les deux
        #affichage du contour convexe
        cv2.line(imageDeBase, start, end, [0, 255, 0], 2)
        cv2.circle(imageDeBase, start, 5, [0, 255, 0], -1)

        #calcul des vecteurs normalises entre le defaut de convexite et le contour convexe
        vecBA = np.array([far[0]-start[0],far[1]-start[1]])
        vecBA = vecBA / np.linalg.norm(vecBA)
        vecBC = np.array([far[0] - end[0], far[1] - end[1]])
        vecBC = vecBC / np.linalg.norm(vecBC)

        #calcul de l'angle entre les vecteurs
        angle = np.arccos(np.dot(vecBA,vecBC))*180/pi
        #on ne considere que les defauts de convexite qui forme un angle assez petit
        if (angle <100) :
            defautDetecte += 1
            #affichage des defauts selectionnes
            cv2.circle(imageDeBase, far, 5, [0, 0, 255], -1)
    # affichage defauts
    # cv2.imshow('fin', imageDeBase)

    convex=False
    if defautDetecte <2:
        convex=True

    return convex


def redimention(imageDeBase):
    # changer la taille d'une image :
    # Choix a verifier : on veut une image qui fait du 500 de largeur
    # et on adapte la hauteur en consequence pour ne pas deformer
    LARGEUR = 500
    height, width, channels = imageDeBase.shape
    facteur = float(LARGEUR) / float(height)
    newInput = cv2.resize(imageDeBase, (0, 0), fx=facteur, fy=facteur)
    return newInput


def segmentation(input): #TODO : mettre la bonne segmentation
    image_gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, 100, 150)

    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    ret, thresh = cv2.threshold(opening, 127, 255, 0)
    return thresh


def etude_classificateur(convexite, dents, jsonPath):
    # ouverture du fichier JSON et mise sous forme de dictionnaire
    with open(jsonPath) as json_data:
        data_dict = json.load(json_data)
        print(data_dict)

    #On stock les resultats sous forme de couple (espece, pourcentage de criteres remplis)
    listeResultat = []

    for cle in data_dict.items(): #on parcours les differents types d'arbres
        espece = cle[0]
        nbrAttribus = len(cle[1])
        nbrPositif = 0
        for cle2 in cle[1].items(): #On parcours les differents attrabuts et leurs valeurs
            if (cle2[0] == "convexe" and cle2[1] == convexite):
                nbrPositif += 1
            if (cle2[0] == "dent" and cle2[1] == dents):
                nbrPositif += 1

        listeResultat.append((espece, float(nbrPositif) / float(nbrAttribus) * 100))
    return listeResultat


def main():
    # input = cv2.imread("base_donnee_feuille/hetre/hetre1.jpg")
    # input = cv2.imread("base_donnee_feuille/hetre/hetre2.jpg")
    # input = cv2.imread("base_donnee_feuille/chene/chene2.jpg")
    # input = cv2.imread("base_donnee_feuille/margousier/margousier2.jpg")
    # input = cv2.imread("base_donnee_feuille/bouleau/bouleau2.jpg")
    input = cv2.imread("base_donnee_feuille/platane/platane1.jpg")


    input = redimention(input)

    thresh = segmentation(input)

    #detection dent :
    dents = detection_dent(thresh, input)

    #detection convexite :
    convexite= feuille_convexe(thresh, input)

    listeResultat = etude_classificateur(convexite, dents, 'arbre.json')
    print(listeResultat)


    cv2.waitKey(0)

main()
