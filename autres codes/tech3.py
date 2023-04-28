import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from skimage.feature import hog
from sklearn import svm

start = time.time()

sequence = 'magasin'
sessions = [['A',10],['B',10]]
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
        img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+'.jpg')
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        fd = hog(img1, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2')
        print('Forme du vecteur caractéristique :',fd.shape)
        hog_features.append(fd)
    labels = np.array(range(s_1[1]))
    clf = svm.SVC()
    hog_features = np.array(hog_features)
    # data_frame = np.hstack((hog_features,labels))
    clf.fit(hog_features,labels)

    
    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        hog_features = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+'.jpg')
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            fd = hog(img2, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2')
            hog_features.append(fd)
        labels = np.array(range(s_2[1])).reshape(s_2[1],1)
        hog_features = np.array(hog_features)

        y_pred = clf.predict(hog_features)
        for y in range(s_2[1]):
            if (y != y_pred[y]):
                error = abs(y-y_pred[y])
                print(s_1, '->',s_2,'\tError for',y,'\tindex = ',y_pred[y],' instead of ',y)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                