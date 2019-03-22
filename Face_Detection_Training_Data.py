# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:37:54 2019

@author: Mithilesh
"""

""" 1-Read the photo using the webcam
    2-Using a predefined classifier named Haarcascade,detect the face and draw a rectangular boundary around the face.
    3-Cut out the Region of interest (ROI) i.e the image bounded by the rectangle by slicing the original image.
    4-Flatten the largest image and store it into a numpy array.
    5-Repeat the above steps some number of times in order to generate training data.
"""
  
#importing neccessary modules    
import cv2
import numpy as np

#Intializing camera(webcam here)
cap=cv2.VideoCapture(0)

#making a skip variable whose purpose is defined in line number 71.
skip=0

#loading haarcascade classifier in order to use its features.
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
 
#making a list here in order to save the largest flatten images as numpy arrays
face_data=[]

#taking an user input ,name of the person whose photo is being taken."
file_name=input("Enter the name of the person whose is being taken : ")
#----------------------------------READING THE IMAGE----------------------------------------------------------------
#Driver Code
while True:
    boolean,frame=cap.read()
    """ From above we get two things,one is a boolean value(either True or False) which tells whether 
        the captured frame is good or not. If the captured image is not good then again start.
    """
    if boolean==False:
        continue
    
    """ Here we use a feature of haarcascade which take three arguments, they are frame,scaling factor and 
        number of neighbours.Like in this example the image is shrinked to 30% of original size 
        & no. of neighnours are 5. 
    """        
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    if len(faces)==0:
        continue
    
    """ Faces will be a list that will have tuples containing four values :- x,y,w,h. i.e (x,y) coordinates
        (starting point of image) and 'w' is width of the image while 'h' is height if the image.
        Since area of the image will be given by w*h here, so higher the value of w*h, larger the image.
        So we will sort the faces here in reverse order using lambda function which will sort it according 
        area i.e w*h which 3rd and 4th value of faces i.e indexed at 2 and 3 respectively.
    """
    
    faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True)
    #----------------------------GIVING BOUNDARY AND SAVING THE ROI--------------------------------------------------------------------    
    for (x,y,w,h) in faces:
        #here we will draw a rectangle around the face.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        # here we crop out the region of interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        #resizing the photo according to the requirement so that detector can work efficiently
        face_section=cv2.resize(face_section,(100,100))
        
        skip+=1
        #here skip is used so that we can store every 10th image that is captured, avoiding the other 9.
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
    #--------------------------------------ROI WORK FINISH-----------------------------------------------------------------
    #showing the image here
    cv2.imshow("Photo",frame)
    cv2.imshow("Region of Interest",face_section)
    
    #making it user friendly so when user press 'q' , the above process will stop
    
    keypressed=cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'):
        break
#--------------------------------------------IMAGE PROCESSING DONE-------------------------------------------------------------
#---------------------------------------SAVING THE DATA INTO NUMPY ARRAY------------------------------------------------------------    
#converting face data into numpy array
face_data=np.asarray(face_data) 

#now reshaping the face data and storing the data of one image into a single row
face_data=face_data.reshape((face_data.shape[0],-1))

#now saving the data as '.npy' file
np.save(file_name+'.npy',face_data)
#-----------------------------------SAVING DONE--------------------------------------------------------
cap.release()
cv2.destroyAllWindows()        
            
 #---------------------------------------XXXXX---------------------------------------------------------       
