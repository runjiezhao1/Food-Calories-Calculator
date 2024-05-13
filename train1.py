import numpy as np        
import os                  
from random import shuffle 
import glob
import cv2
from cnn_model import get_model

path = './Myfood/images/train_set'
IMG_SIZE = 256
LR = 0.001
#Fruits_dectector-{}-{}.model
MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, 'Allconv-basic')
no_of_fruits=14
percentage=0.3
no_of_images=100

def create_train_data(path):
    training_data = []
    folders=os.listdir(path)[0:no_of_fruits]
    print("folders ",folders)
    for i in range(len(folders)):
        label = [0 for i in range(no_of_fruits)]
        label[i] = 1
        print("folders",folders[i])
        k=0
        for j in glob.glob(path+"\\"+folders[i]+"\\*.jpg"):            
            if(k==no_of_images):
                break
            k=k+1
            img = cv2.imread(j)
            print(j)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    return training_data,folders

training_data,labels=create_train_data(path)
print("labels ", labels)
size=int(len(training_data)*percentage)
train = training_data[:-size]
test=training_data[-size:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print("Y ", Y)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]

model=get_model(IMG_SIZE,no_of_fruits,LR)

model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model_save_at=os.path.join("model",MODEL_NAME)
model.save(model_save_at)
print("Model Save At",model_save_at)