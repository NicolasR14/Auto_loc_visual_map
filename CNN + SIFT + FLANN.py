from tensorflow.keras.applications import VGG16
from tensorflow.python.client import device_lib
from keras import backend as K
import cv2
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import scipy
import matplotlib.pyplot as plt
import time


#première couche du réseau : on utilise la couche de VGG16 pour transformer l'iamge.
vgg_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(480, 640, 3))


output_layer = vgg_base.layers[1].output
input_layer = vgg_base.layers[0].input
get_layer_output= K.function([input_layer], [output_layer])


#fonction chargée de récupérer les points clés et descripteurs pour une image passée en paramètre
def extract_kp_des(img):
    img = np.expand_dims(img,axis=0)
    img = get_layer_output(img)
    img = np.array(img)[0,:,:,:]

    #application d'un filtre laplacien
    img = scipy.ndimage.filters.laplace(img)[0,:]

    #récupérer la distance euclidienne
    img_mag = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(img)):
        for j in range(len(img[i])):
            img_mag[i][j] = np.linalg.norm(img[i][j])

    img = cv2.normalize(img_mag, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return sift.detectAndCompute(img,None) #return kp, des

#définition du SIFT
nb_kp = 500
sift = cv2.SIFT_create(nb_kp)

sequence = 'magasin'
sessions = [['A',11],['B',11]]
print("Test sur la sequence '"+sequence+"'")
# paramètres de FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

results = []
start = time.time()
error_tot = 0
extension = '.jpg'
ratio_lowe = 0.7

for s_1 in sessions:
    #s_1 est la session qui sert à créer la carte visuelle
    s_others = [s_o for s_o in sessions if s_o != s_1] #autres sessions que s_1
    #charge les images de s_1
    s_1_map = [] #carte visuelle

    for index in range(s_1[1]):
        index_img = "{:0>2}".format(index)
        try :
            img1 = cv2.imread('./images/'+sequence+s_1[0]+'-'+str(index_img)+extension)
            # keypoints et descripteurs avec SIFT
            kp1, des1 = extract_kp_des(img1)
            s_1_map.append({'kp':kp1,'des':des1})
        except :
            print("'./images/"+sequence+s_1[0]+'-'+str(index_img)+extension + "' n'existe pas")
            continue

    for s_2 in s_others : #s_2 : session d'où les nouvelles images proviennent
        for index in range(s_2[1]):
            index_img = "{:0>2}".format(index)
            try :
                img2 = cv2.imread('./images/'+sequence+s_2[0]+'-'+str(index_img)+extension)
                
                # keypoints et descripteurs avec SIFT
                kp2, des2 = extract_kp_des(img2)
            except :
                print("'./images/"+sequence+s_2[0]+'-'+str(index_img)+extension + "' n'existe pas")
                continue

            #on teste le matching de correspondances pour chaque élément de la carte
            proba = []
            for img1 in s_1_map:
                matches = flann.knnMatch(img1['des'],des2,k=2)
                goodmatches = 0
                 # test selon le ratio de Lowe
                for i,(m,n) in enumerate(matches):
                    if m.distance < ratio_lowe*n.distance:
                        goodmatches += 1
                proba.append(goodmatches)

            #tri selon le nombre de matchs
            if sum(proba) != 0:
                proba = [p/sum(proba) for p in proba]
                max_value = max(proba)
                max_index = proba.index(max_value)
                error = abs(max_index-index)
                if (error > 0 ):
                    print(s_1, '->',s_2,'\tErreur pour',index_img,'\tindice =',max_index,'au lieu de',index)
                    error_tot += error
            else :
                print("Impossible de predire pour",index_img)

end = time.time()
print('\nErreur('+','.join([s[0] for s in sessions])+') = ',error_tot)
print("\nTemps d'execution : ", "{:.3f}s".format(end - start))