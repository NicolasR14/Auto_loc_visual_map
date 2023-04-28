import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

sequence = 'visages'
sessions = [['A',10],['B',10]]
nb_kp = 500
n_neigh = 1
extension = '.jpg'
print('Test pour la sequence',sequence)

error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    # intialisation de KNN
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    # detecteur SIFT initial
    sift = cv2.SIFT_create(nb_kp)
    labels = np.array([])
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        try :
            img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
            # keypoints et descripteurs avec SIFT
            kp1, des1 = sift.detectAndCompute(img1,None) #des1.shape = nb_kp * 128
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+ extension + "' n'existe pas")
            continue
        des1 = np.array(des1)
        s_1_map.append(des1)
        labels = np.append(labels,index*np.ones(len(des1)))
    s_1_map = np.array(s_1_map)
    s_1_map = np.vstack(s_1_map)
    neigh.fit(s_1_map, labels)

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        X_test = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            try:
                img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
                kp2, des2 = sift.detectAndCompute(img2,None)
            except :
                print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+ extension + "' n'existe pas")
                continue
            des2 = np.array(des2)
            predi = neigh.predict(des2) #prédiction pour chaque point-clé
            count_predi = np.zeros(s_1[1]) #compteur de classification
            for p in predi:
                count_predi[int(p)] += 1

            #tri selon le nombre de classifications
            proba = [c/sum(count_predi) for c in count_predi]
            max_value = max(proba)
            max_index = proba.index(max_value)
            error = abs(max_index-index)
            if (error > 0 ):
                print(s_1, '->',s_2,'\tErreur pour',index_img,'\tindice =',max_index,'au lieu de',index)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                