import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

target_size=(128,128)
def load_images(dir,target_size):
    files=os.listdir(dir)
    images=[]
    for f in files:
        file_name=os.path.join(dir,f)
        img=cv2.imread(file_name)
        if img is not None:
            resized_img = cv2.resize(img, target_size)
            images.append(resized_img)
    return images

fall_images=load_images(r"C:\Users\Diya\OneDrive\Documents\College\Projects\Fall\fall_dataset\images\Fall",target_size)
no_fall_images=load_images(r"C:\Users\Diya\OneDrive\Documents\College\Projects\Fall\fall_dataset\images\NoFall",target_size)

fall_images_label=[1]*len(fall_images)
nofall_images_label=[0]*len(no_fall_images)

images=np.array(fall_images+no_fall_images)
labels=np.array(fall_images_label+nofall_images_label)

images=images/255

X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=0)

