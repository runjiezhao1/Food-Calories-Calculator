#image_segment
import cv2
import numpy as np
import os

import torch
from PIL import Image
from torchvision import transforms

class Label_encoder:
    def __init__(self, labels):
        labels = list(set(labels))
        self.labels = {label: idx for idx, label in enumerate(classes)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels[label]

def classify_image(image_path, model, label_encoder, device):
    # Load and preprocess the input image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)

    # Get predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()

    # Map index to class name
    predicted_label = label_encoder.get_label(predicted_idx)

    return predicted_label

model_classifier = torch.load("./model/food_classification.pt")
image_path = "dataset/images/test_set/apple/apple (1).jpg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = open("./dataset/classes.txt", 'r').read().splitlines()
label_encoder = Label_encoder(classes)

predicted_label = classify_image(image_path, model_classifier, label_encoder, device)
print("Predicted Label:", predicted_label)

#get the maskrcnn and image volume here
def getAreaOfFood(img1):
    data=os.path.join(os.getcwd(),"images")
    if os.path.exists(data):
        print('folder exist for images at ',data)
    else:
        os.mkdir(data)
        print('folder created for images at ',data)

    cv2.imwrite('{}\\1 original image.jpg'.format(data),img1)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}\\2 original image BGR2GRAY.jpg'.format(data),img)
    img_filt = cv2.medianBlur( img, 5)
    cv2.imwrite('{}\\3 img_filt.jpg'.format(data),img_filt)
    img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    cv2.imwrite('{}\\4 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #make change here
    cv2.drawContours(img_th, contours, -1, (0,255,0), 3)
    #print("inside get area of food")
	# find contours. sort. and find the biggest contour. the biggest contour corresponds to the plate and food.
    mask = np.zeros(img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
    cv2.imwrite('{}\\5 mask.jpg'.format(data),mask)
    img_bigcontour = cv2.bitwise_and(img1,img1,mask = mask)
    cv2.imwrite('{}\\6 img_bigcontour.jpg'.format(data),img_bigcontour)

	# convert to hsv. otsu threshold in s to remove plate
    hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
    cv2.imwrite('{}\\7 hsv_img.jpg'.format(data),hsv_img)
    h,s,v = cv2.split(hsv_img)
    mask_plate = cv2.inRange(hsv_img, np.array([0,0,50]), np.array([200,90,250]))
    cv2.imwrite('{}\\8 mask_plate.jpg'.format(data),mask_plate)
    mask_not_plate = cv2.bitwise_not(mask_plate)
    cv2.imwrite('{}\\9 mask_not_plate.jpg'.format(data),mask_not_plate)
    food_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)
    cv2.imwrite('{}\\10 food_skin.jpg'.format(data),food_skin)

	#convert to hsv to detect and remove skin pixels
    hsv_img = cv2.cvtColor(food_skin, cv2.COLOR_BGR2HSV)
    cv2.imwrite('{}\\11 hsv_img.jpg'.format(data),hsv_img)
    skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
    cv2.imwrite('{}\\12 skin.jpg'.format(data),skin)
    not_skin = cv2.bitwise_not(skin); #invert skin and black
    cv2.imwrite('{}\\13 not_skin.jpg'.format(data),not_skin)
    food = cv2.bitwise_and(food_skin,food_skin,mask = not_skin) #get only food pixels
    cv2.imwrite('{}\\14 food.jpg'.format(data),food)

    food_bw = cv2.cvtColor(food, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}\\15 food_bw.jpg'.format(data),food_bw)
    food_bin = cv2.inRange(food_bw, 10, 255) #binary of food
    cv2.imwrite('{}\\16 food_bw.jpg'.format(data),food_bin)

	#erode before finding contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erode_food = cv2.erode(food_bin,kernel,iterations = 1)
    cv2.imwrite('{}\\17 erode_food.jpg'.format(data),erode_food)

	#find largest contour since that will be the food
    img_th = cv2.adaptiveThreshold(erode_food,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\18 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_food = np.zeros(food_bin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_food, [largest_areas[-2]], 0, (255,255,255), -1)
    cv2.imwrite('{}\\19 mask_food.jpg'.format(data),mask_food)

	#dilate now
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask_food2 = cv2.dilate(mask_food,kernel2,iterations = 1)
    cv2.imwrite('{}\\20 mask_food2.jpg'.format(data),mask_food2)
    food_final = cv2.bitwise_and(img1,img1,mask = mask_food2)
    cv2.imwrite('{}\\21 food_final.jpg'.format(data),food_final)

	#find area of food
    img_th = cv2.adaptiveThreshold(mask_food2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\22 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours, key=cv2.contourArea)
    food_contour = largest_areas[-2]
    food_area = cv2.contourArea(food_contour)


	#finding the area of skin. find area of biggest contour
    skin2 = skin - mask_food2
    cv2.imwrite('{}\\23 skin2.jpg'.format(data),skin2)
	#erode before finding contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    skin_e = cv2.erode(skin2,kernel,iterations = 1)
    cv2.imwrite('{}\\24 skin_e .jpg'.format(data),skin_e )
    img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\25 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_skin = np.zeros(skin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    index = -2
    if len(largest_areas) <= 1:
        index = -1
    cv2.drawContours(mask_skin, [largest_areas[index]], 0, (255,255,255), -1)
    cv2.imwrite('{}\\26 mask_skin.jpg'.format(data),mask_skin)


    skin_rect = cv2.minAreaRect(largest_areas[index])
    box = cv2.boxPoints(skin_rect)
    box = np.int0(box)
    mask_skin2 = np.zeros(skin.shape, np.uint8)
    cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)
    cv2.imwrite('{}\\27 mask_skin2.jpg'.format(data),mask_skin2)

    pix_height = max(skin_rect[1])
    pix_to_cm_multiplier = 5.0/pix_height
    skin_area = cv2.contourArea(box)


    return food_area,food_bin ,food_final,skin_area, food_contour, pix_to_cm_multiplier

#caleries
import cv2
import numpy as np
food_dict = {"apple":1,"bibimbap":2,"bulgogi":3,"chocolate_cake":4,"fried_egg":5,"hamburger":6,"jajangmyeon":7,"kimbap":8,"kimchi_stew":9,"pizza":10,"ramen":11,"sandwich":12,"steak":13,"sushi":14}
#density - gram / cm^3
density_dict = { 1:0.9, 2:0.94, 3:0.641,  4:0.641,5:0.513, 6:0.482,7:0.481,8:0.234,9:0.497,10:0.677,11:0.456,12:0.765,13:0.343,14:0.245}
#kcal
calorie_dict = { 1:0.00052, 2:0.89, 3:0.41,4:0.16,5:0.40,6:0.47,7:0.18, 8:0.23,9:0.18,10:0.90,11:0.87,12:0.45,13:0.80,14:0.10}
#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getCalorie(label, volume): #volume in cm^3
	calorie = calorie_dict[int(label)]
	density = density_dict[int(label)]
	mass = volume*density*1.0
	calorie_tot = (calorie/100.0)*mass
	return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, food_contour):
	area_food = (area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
    #sphere
	if label == 1 or label == 5 or label == 7 or label == 6 or label == 12 or label == 13 or label == 14: 
		radius = np.sqrt(area_food/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		#print (area_food, radius, volume, skin_area)

    #column
	if label == 2 or label == 4 or label == 3 or label == 8 or label == 9 or label == 10 or label == 1: 
		food_rect = cv2.minAreaRect(food_contour)
		height = max(food_rect[1])*pix_to_cm_multiplier
		radius = area_food/(2.0*height)
		volume = np.pi*radius*radius*height

	if (label==4 and area_food < 30) : 
		volume = area_food*0.5 

	return volume

def calories(result,img):
    img_path =img
    food_areas,final_f,areaod,skin_areas, food_contours, pix_cm = getAreaOfFood(img_path)
    volume = getVolume(result, food_areas, skin_areas, pix_cm, food_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    food_volumes=volume
    food_calories=cal
    food_calories_100grams=cal_100
    food_mass=mass
    return food_calories

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

def get_model(IMG_SIZE,no_of_foods,LR):
	try:
		tf.reset_default_graph()
	except:
		print("tensorflow")
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)


	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, no_of_foods, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	return model

import os
import cv2
import numpy as np

IMG_SIZE = 256
LR = 1e-3
no_of_foods=14

MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, 'Allconv-basic')

model_save_at=os.path.join("model",MODEL_NAME)

model_mask=get_model(IMG_SIZE,no_of_foods,LR)

model_mask.load(model_save_at)
labels=list(np.load('labels.npy'))
print("labels ",labels)
img=cv2.imread(image_path)
img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
model_out=model_mask.predict([img1])
result=np.argmax(model_out)
result = food_dict[predicted_label]
print("result", result)
cal=round(calories(result,img),2)

import matplotlib.pyplot as plt
plt.imshow(img)
plt.title('{}({}kcal)'.format(predicted_label,cal))
plt.axis('off')
plt.show()