import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

start = time.time()

sequence = 'legumes'
sessions = [['A',10],['B',10]]
nb_kp = 50
n_clusters = 200

error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    # Initiate SIFT detector
    sift = cv2.SIFT_create(nb_kp)
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+'.jpg')
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None) #des1.shape = nb_kp * 128

        if (len(des1) > nb_kp): #dans le cas où plusieurs kp ont le même score
            des1 = des1.tolist()
            while(len(des1)>nb_kp):
                des1.pop()
            des1 = np.array(des1)

        des1 = des1.flatten()
        s_1_map.append(des1) #s_1_map.shape = nb_kp * 128 * s_1[1] = nombre de keypoints par image * 128 * nombre d'images

    s_1_map = np.array(s_1_map)
    labels = np.array(range(s_1[1]))

    components = 10
    pca = PCA(n_components=components, svd_solver='randomized',whiten=True).fit(s_1_map)
    n_samples, h, w = s_1[1],nb_kp,128
    X_train_pca = pca.transform(s_1_map)
    print(X_train_pca.shape)
    neigh_pca = KNeighborsClassifier(n_neighbors=1)
    neigh_pca.fit(X_train_pca, labels)

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        X_test = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+'.jpg')
            kp2, des2 = sift.detectAndCompute(img2,None)
            if (len(des2) > nb_kp): #dans le cas où plusieurs kp ont le même score
                des2 = des2.tolist()
                while(len(des2)>nb_kp):
                    des2.pop()
                des2 = np.array(des2)
            des2 = des2.flatten()
            X_test.append(des2)
        X_test = np.array(X_test)
        X_test_pca = pca.transform(X_test)
        
        predi = neigh_pca.predict(X_test_pca)
        for y in range(s_2[1]):
            if (y != predi[y]):
                error = abs(y-predi[y])
                print(s_1, '->',s_2,'\tError for',y,'\tindex = ',predi[y],' instead of ',y)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                