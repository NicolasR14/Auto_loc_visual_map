import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()

sequence = 'parc'
sessions = [['A',10],['B',10]]
extension = '.jpg'
ratio_lowe = 0.7
print("Test sur la sequence '"+sequence+"'")
# paramètres de Flann
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 

flann = cv2.FlannBasedMatcher(index_params,search_params)

results = []

error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    # Initialisation de Brisk
    brisk = cv2.BRISK_create()
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        try :
            img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
            # keypoints et descripteurs avec Brisk
            kp1, des1 = brisk.detectAndCompute(img1,None)
            s_1_map.append({'kp':kp1,'des':des1})
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+extension + "' n'existe pas")
            continue

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            try :
                img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
                
                # keypoints et descripteurs avec Brisk
                kp2, des2 = brisk.detectAndCompute(img2,None)
            except :
                print("'./images/"+sequence+s_2[0]+'-'+str(index_img)+extension + "' n'existe pas")
                continue

            #on teste le matching de correspondances pour chaque élément de la carte
            proba = []
            for img1 in s_1_map:
                matches = flann.knnMatch(np.float32(img1['des']),np.float32(des2),k=2)
                goodmatches = 0
                # ratio de Lowe
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        goodmatches += 1
                proba.append(goodmatches)

            #tri selon le nombre de matchs
            if (sum(proba) != 0):
                proba = [p/sum(proba) for p in proba]
                max_value = max(proba)
                max_index = proba.index(max_value)
                error = abs(max_index-index)
                if (error > 0 ):
                    print(s_1, '->',s_2,'\tErreur pour',index_img,'\tindice = ',max_index,'au lieu de',index)
                    error_tot += error
            else :
                print("Impossible de predire pour",index_img)
end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                