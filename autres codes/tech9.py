import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()

sequence = 'neige'
sessions = [['A',10],['B',10]]
nb_kp = 200

error_tot = 0
for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle
    akaze = cv2.AKAZE_create() 
    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+'.jpg')
        # find the keypoints and descriptors with AKAZE
        kpts1, desc1 = akaze.detectAndCompute(img1, None)

        desc1 = desc1.flatten()
        s_1_map.append(desc1)

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        X_test = []
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+'.jpg')
            # find the keypoints and descriptors with AKAZE
            kpts2, desc2 = akaze.detectAndCompute(img2, None)
            desc2 = desc2.flatten()
            distance_map = []
            for d in s_1_map:
                distance = np.linalg.norm(d-desc2)
                distance_map.append(distance)
            min_distance = min(distance_map)
            min_index = distance_map.index(min_distance)
            error = abs(min_index-index)
            if (error > 0 ):
                print(s_1, '->',s_2,'\tError for',index_img,'\tindex = ',min_index,' instead of ',index)
                error_tot += error

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))
            
            


                