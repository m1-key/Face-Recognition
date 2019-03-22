# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 01:08:18 2019

@author: Mithilesh
"""

""" Recognize the face by using the K-Nearest Neighbours Algorithm.
    1-Load the data of every person stored as numpy array.
        Here X will be the information about the image.
        Y will be the id allocated to each person.
    2- Read the image from the webcam.
    3- Crop the region of interest.
    4- Using KNN , predict the outcome id.
    5- Allocate name to the outcome id.
    6- Display the name of the user with a rectangular boundary drawn over the image.
"""
import cv2
import numpy as np
import os

#-------------------------------KNN STARTS-----------------------------------------

#First we define distance function
def dist(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())
    
#Now defining KNN Algorithm
def KNN(train,test,k=5):
    #X will have all the except last one as last one will contain the ID which is allocated to every image
    X=train[:,:-1]
    Y=train[:,-1]
    #creating a list which will save the value of distance between test and training point and will save id also.
    vals=[]
    
    for i in range(X.shape[0]):
        vals.append((dist(test,X[i]),Y[i]))
    
    """Now we will sort the vals on the basis of distance and will take only first 'k' values from it as they
       will be the nearest ones."""  
    vals=sorted(vals)
    vals=vals[:k]
    
    #Now we will find which class id is more closer to the test image.
    new_vals=np.unique(vals[1],return_counts=True)
    max_ind=np.argmax(new_vals[1])
    prediction=new_vals[0][max_ind]
    return(int(prediction))
    
#--------------------------------KNN ENDS----------------------------------------------------------
    
#----------------------------DATA PREPARATION------------------------------------------------------

#This will store the all the training data
face_data=[]

#This will an array which will save class ids.
labels=[]

#defining class id which will allocated to every image.
class_id=0

#A dcitionary is defined here which will treat class id as key and its value will be the name of the file
names={}

for file in os.listdir():
    if file.endswith('.npy'):
        #Loading the training data one at a time
        data_item=np.load(file)
        face_data.append(data_item)
        """Now creating an array which will give a class id to the data item. Since data item will have multiple
           images, therefore we created an array which will give same class id to all the images belonging to
           same data item.
        """
        target=np.ones((data_item.shape[0],))*class_id
        labels.append(target)
        
        """Allocating file name to the class id. As last four characters will be'.npy', so we sliced it out
           to get a name like 'ABCXYZ' from 'ABCXYZ.npy.
        """
        names[class_id]=file[:-4]
        #incrementing class id after each iteration
        class_id+=1

#Now converting face data and labels into a specific type so that they can be merged together into a single array
face_data=np.concatenate(face_data,axis=0)
face_label=np.concatenate(labels,axis=0).reshape((-1,1))

#Merging them into a single array. Label is the last column.
training_data=np.append(face_data,face_label,axis=1)

#---------------------------------------DATA PREPARATION ENDS----------------------------------------------

#-------------------------------READING THE IMAGE AND MAKING PREDICTION STARTS-------------------------------------

#Its documentation is same as written in Face_Detection_Training_Data

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    boolean,frame=cap.read()
    if boolean==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for (x,y,w,h) in faces:
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        
        """Now we will flatten this face section into a single row as our each training data is stored
           into a single row. We flatten this into a single row so that dist() function of KNN part can
           be executed as both train and test will have same shape so (x1-x2) will be executed.
        """
        face_section=face_section.flatten()
        
        #Now calling the KNN to predict the output
        output=KNN(training_data,face_section,k=5)
        predicted_name=names[output]
        
        #Now giving the rectangular border with the predicted name wriiten above it to the image using putText()
        cv2.putText(frame,predicted_name,(x+10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    cv2.imshow("Prediction ",frame)

    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'):
        break
#------------------------------------------PREDICTION OVER---------------------------------------------    
cap.release()
cv2.destroyAllWindows()
#-----------------------------------------XXXXXX-------------------------------------------------------