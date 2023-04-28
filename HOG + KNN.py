import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier

start = time.time()

sequence = 'visages'
sessions = [['A',10],['B',10]]
extension = '.jpg'
error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    ppc = 16
    hog_features = []
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
        try:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            fd = hog(img1, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2') #extraction des caractéristiques avec HOG
            hog_features.append(fd)
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+extension + "' n'existe pas")
            continue
    labels = np.array(range(s_1[1]))
    neigh = KNeighborsClassifier(n_neighbors=1)
    hog_features = np.array(hog_features)
    neigh.fit(hog_features, labels)

    
    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        hog_features = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
            try:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                fd = hog(img2, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2') #extraction des caractéristiques avec HOG
                hog_features.append(fd)
            except :
                print("'./images/"+sequence+s_2[0]+'-'+str(index_img)+extension + "' n'existe pas")
                continue
        labels = np.array(range(s_2[1])).reshape(s_2[1],1)
        hog_features = np.array(hog_features)
        
        predi = neigh.predict(hog_features) #prédiction avec KNN
        for y in range(s_2[1]):
            if (y != predi[y]):
                error = abs(y-predi[y])
                print(s_1, '->',s_2,'\tErreur pour',y,'\tindice =',predi[y],'au lieu de',y)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                