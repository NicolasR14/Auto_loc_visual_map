import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()

sequence = 'parc'
sessions = [['A',10],['B',10]]
m_to_count = 50
extension = '.jpg'
print("Test sur la sequence '"+sequence+"'")
#paramètres de BFMatcher

results = []

error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    # Initialisation de Brisk
    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    orb = cv2.ORB_create()
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        try :
            img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
            # keypoints et descripteurs avec Brisk
            kp1, des1 = orb.detectAndCompute(img1,None)
            clusters = np.array([des1])
            bf.add(clusters)
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+extension + "' n'existe pas")
            continue

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            try :
                img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
                
                # keypoints et descripteurs avec Brisk
                kp2, des2 = orb.detectAndCompute(img2,None)
            except :
                print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+extension + "' n'existe pas")
                continue
            #on teste le matching de correspondances pour chaque élément de la carte
            # Match des valeurs des descripteurs avec Brute Force Matcher
            matches = bf.match(des2)
            # Tri selon leur distances
            matches = sorted(matches, key = lambda x:x.distance)
            
            index_predict = []
            for i in range(len(matches)):
                if len(index_predict) > m_to_count:
                    break
                index_predict.append(matches[i].imgIdx)

            max_index = max(index_predict,key=index_predict.count)
            if max_index != index :
                error_tot += abs(max_index-index)
                print(s_1, '->',s_2,'\tErreur pour',index_img,'\tindice =',max_index,' au lieu de ',index)

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                