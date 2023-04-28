import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.feature import hog

start = time.time()

sequence = 'visages'
sessions = [['A',10],['B',10]]
error_tot = 0
extension = '.jpg'
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    ppc = 16
    hog_features = []
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        try :
            img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            fd = hog(img1, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2') #extraction des caractéristiques avec HOG
            s_1_map.append(fd)
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+ extension + "' n'existe pas")
            continue
    
    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        hog_features = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            try:
                img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                fd = hog(img2, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2') #extraction des caractéristiques avec HOG
            except :
                print("'./images/"+sequence+s_2[0]+'-'+str(index_img)+extension + "' n'existe pas")
                continue
            #tri selon distance euclidienne
            distance_map = []
            for h in s_1_map:
                distance = np.linalg.norm(h-fd)
                distance_map.append(distance)
            min_distance = min(distance_map)
            min_index = distance_map.index(min_distance)
            error = abs(min_index-index)
            if (error > 0 ):
                print(s_1, '->',s_2,'\tErreur pour',index_img,'\tindice =',min_index,'au lieu de',index)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                