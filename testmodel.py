import cv2 
import numpy as np
from matplotlib import pyplot as plt  
import os
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten
from keras.datasets import mnist

fgbg = cv2.createBackgroundSubtractorMOG2()
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
#loading weights
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

print(loaded_model.summary())  

font= cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (70,70)
fontScale= 2
fontColor= (255,255,255)
lineType =2     
  

  #deploying the model
r=(110, 126, 229, 223)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,image = cap.read()
    copy=image.copy()
    image = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    im_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
 
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)
    cv2.imshow("edge",image)
    cv2.imshow("frame",skin_ycrcb)
    cv2.imwrite("cam.jpg", skin_ycrcb)
    imgs=cv2.imread('cam.jpg')
    imgs = cv2.resize(imgs,(64,64)) 
    imgs=np.reshape(imgs, (1,64,64,3))
    imgs =imgs.astype('float32')/255
    val=loaded_model.predict(imgs,verbose=0)
    val=np.reshape(val,(16,1))
    #val=nqp.round(val)
    if(val[0]>0.60):
        num='one'
    elif(val[1]>0.60):
        num='two'
    elif(val[2]>0.60):
	    num='three'
    elif(val[3]>0.60):
	    num='four'
    elif(val[4]>0.60):
	    num='five'
    elif(val[5]>0.60):
	    num='six'
    elif(val[6]>0.60):
	    num='seven'
    elif(val[7]>0.60):
	    num='eight'
    elif(val[8]>0.60):
	    num='nine'
    elif(val[9]>0.60):
	    num='A'
    elif(val[10]>0.60):
	    num='C'
    elif(val[11]>0.60):
	    num='I'
    elif(val[12]>0.60):
	    num='K'
    elif(val[13]>0.60):
	    num='L'
    elif(val[14]>0.60):
	    num='O'
    elif(val[15]>0.60):
	    num='Y'
    else:
        num="no idea"
    cv2.putText(copy,num, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

    cv2.imshow("final",copy)
   
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break