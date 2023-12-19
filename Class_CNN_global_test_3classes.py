# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:11.287306Z","iopub.execute_input":"2023-05-29T07:00:11.288150Z","iopub.status.idle":"2023-05-29T07:00:11.322350Z","shell.execute_reply.started":"2023-05-29T07:00:11.288104Z","shell.execute_reply":"2023-05-29T07:00:11.321072Z"}}
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:12:31 2023

@author: ippop
"""

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:11.324610Z","iopub.execute_input":"2023-05-29T07:00:11.325303Z","iopub.status.idle":"2023-05-29T07:00:21.139367Z","shell.execute_reply.started":"2023-05-29T07:00:11.325266Z","shell.execute_reply":"2023-05-29T07:00:21.138203Z"}}
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
tf.config.list_physical_devices('GPU')


import cv2
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold,StratifiedShuffleSplit

import matplotlib.pyplot as plt
 
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay



#from pydicom import dcmread
import pandas as pd
import platform
from IPython.display import clear_output

OS=platform.system()
if (OS == 'Windows)'):
    data_dir =r'C:\DP\DATI_CUORE_LOC'
else:
    data_dir='C:\DP\DATI_CUORE_LOC\DL_DATA\DICOM_CLASSES'

img_height , img_width = 40, 40 # image dimension at model input
slices=3
seq_len = 10
 
#classes = ["Borderline", "Mild", "Moderate", "Normal", "Severe"]
#classes = ["NormalHomogeneous","NormalHeterogeneous", "PatHeterogeneous", "PatHomogeneous"]
#classes = ["Homogeneous","Heterogeneous"]
classes = ["Normal","IronOverload",'Severe']

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:21.141343Z","iopub.execute_input":"2023-05-29T07:00:21.142677Z","iopub.status.idle":"2023-05-29T07:00:21.172144Z","shell.execute_reply.started":"2023-05-29T07:00:21.142628Z","shell.execute_reply":"2023-05-29T07:00:21.170950Z"}}
# image crop function
def frames_crop(frames,size):
    
    image=np.asarray(frames,dtype='uint16') # we have a 3x10 array
    
    #crop size
    image1=image[0]
    (dx,dy)=image1.shape
    min_x=round(dx/4.)  
    max_x=round(3.*dx/4.)
    min_y=round(dy/4.)
    max_y=round(3.*dy/4.)

    crop=image1[min_x:max_x,min_y:max_y]
    crop=cv2.resize(crop,size)
    
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # imgplot = plt.imshow(crop,cmap='gray')
    # plt.show()
    
    crop=np.expand_dims(crop,axis=2)
    
    cropped_frames=[]
    cropped_frames.append(crop)
    
    for i in range(1, len(frames)):
        image=frames[i]
        crop=image[min_x:max_x,min_y:max_y]
        crop=cv2.resize(crop,size)
        # imgplot = plt.imshow(crop,cmap='gray')
        # plt.show()
        crop=np.expand_dims(crop,axis=2)
        cropped_frames.append(crop)
     
    return cropped_frames
 
def image_crop(image,crop_win,int_size):
    #crop_win = (left_x,left_y,right_x,right_y)
    (dx,dy)=image.shape
    #find window
    min_x=round(crop_win[0])  
    max_x=round(crop_win[1])
    min_y=round(crop_win[2])
    max_y=round(crop_win[3])

    crop=image[min_x:max_x,min_y:max_y]
    crop=cv2.resize(crop,int_size)
     
    return crop   
 
    
 
def frames_load(frames,size):
     
     image=np.asarray(frames,dtype='uint16') # we have a 3x10 array
    
     img=image[0]  # load first image
     img=cv2.resize(img,size) #resize first image
     img=np.expand_dims(img,axis=2) #add time dimension
     
     loaded_frames=[]
     loaded_frames.append(img)
     
     for i in range(1, len(frames)):
         img=image[i]
         img=cv2.resize(img,size)
         img=np.expand_dims(img,axis=2)
         loaded_frames.append(img)
      
     return loaded_frames


 
def create_data(input_dir):
    X = []
    Y = []
     
    classes_list = os.listdir(input_dir)  # classes directories 
    print(classes_list)
     
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c)) # patient's directories
        for f in files_list:
            image_list = os.listdir(os.path.join(os.path.join(input_dir, c), f)) # image files
            frames = []
            count = 0
            print(f)
            # find slice order
            
            dcm_file1=os.path.join(os.path.join(os.path.join(input_dir, c), f),image_list[0])
            dcm_file2=os.path.join(os.path.join(os.path.join(input_dir, c), f),image_list[2*seq_len])
            dcm_img1=dcmread(dcm_file1)
            dcm_img2=dcmread(dcm_file2)
            iP1=dcm_img1.ImagePositionPatient #image position first slice
            iP2=dcm_img2.ImagePositionPatient #image position last slice
            iP =np.asarray([0,0,1000]) # patient's head" :-)
            dist1 = np.linalg.norm(iP1-iP)
            dist2 = np.linalg.norm(iP2-iP)
            if (dist1 < dist2):
                # first slice basal, last apical
                while count < 3*seq_len: 
                    dcm_file=os.path.join(os.path.join(os.path.join(input_dir, c), f),image_list[count])
                    dcm_img=dcmread(dcm_file)
                    image=dcm_img.pixel_array
                    frames.append(image)
                    count += 1
            else:
                # first slice apical, last basal
                #print('reordering patient ',f)
                for s in range(2,-1,-1):
                    count1=0
                    while count1 < seq_len: 
                        dcm_file=os.path.join(os.path.join(os.path.join(input_dir, c), f),image_list[s*seq_len+count])
                        dcm_img=dcmread(dcm_file)
                        image=dcm_img.pixel_array
                        frames.append(image)
                        count1 += 1
                
            cropped_frames=frames_load(frames,(256, 256))
            
            
            X.append(cropped_frames)
             
            y = [0]*len(classes)
            y[classes.index(c)] = 1
            Y.append(y)
     
       
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:21.175550Z","iopub.execute_input":"2023-05-29T07:00:21.176046Z","iopub.status.idle":"2023-05-29T07:00:21.192015Z","shell.execute_reply.started":"2023-05-29T07:00:21.176007Z","shell.execute_reply":"2023-05-29T07:00:21.190946Z"}}
def createAugmentedDataShift(Xor1,Yor1,inc):
    
    Xor=Xor1.copy()
    Yor=Yor1.copy()
    
    X = []
    Y = []
    
    dataSize=Xor.shape[0] #patients
    height=Xor.shape[1] #slices*rows
    columns=Xor.shape[2] #columns 
    seqLen=Xor.shape[3]  #frames
    
    rng = np.random.default_rng(seed=42)
    for i in range(dataSize):
        # add original data
        orXData = Xor[i,:,:,:]
        orYData = Yor[i,:]
        X.append(orXData)
        Y.append(orYData)
        #print('copy image')
        #imgplot = plt.imshow(orXData[:,:,0],cmap='gray')
        #plt.show()
        
        #print('append ',i,len(X))
        # add augmented data
        for j in range(inc-1):
            #random shift
            r=int(tf.random.uniform([], minval=0,maxval=1)) #0/1 value
            CurrentImg=Xor[i,:,:,:] # original data
            augImg = np.zeros([height,columns,seqLen],dtype='uint16')
            
            #r=1
            if(True):
                #print(' perform shift ')
                shift1 = int(tf.random.uniform([], minval=-4,maxval=4))   # vertical shift
                shift2 = int(tf.random.uniform([], minval=-2,maxval=2))   # horizontal shift
                sigma=tf.random.uniform([], minval=1.0,maxval=20.0)
                for f in range(seqLen):
                    img1 = CurrentImg[:,:,f]
                    n1=sigma*np.random.randn(height,columns)
                    n2=sigma*np.random.randn(height,columns)
                    img2 = np.sqrt((img1.copy()+n1)**2+n2**2) 
                    img3 = np.roll(img2.copy(), shift1,axis=1) # round shift over rows
                    img4 = np.roll(img3.copy(), shift2,axis=0) # round shift over rows
                    augImg[:,:,f]=np.reshape(img4.copy(),(height,columns)) 
            # if(r==1):   
            #     #print(' add bias ')
            #     bias = int(tf.random.uniform([], minval=0,maxval=50.0))
            #     for s in range(segments):
            #         for f in range(seqLen):    
            #             img1 = CurrentImg[s,:,:,f]
            #             img2 = img1.copy()+bias # add bias
            #             augImg[s,:,:,f]=img2.copy() 
            # if(True):   
            #       newSeg=np.random.permutation(np.arange(segments))
            #       for s in range(segments):
            #           for f in range(seqLen):    
            #               img1 = augImg[s,:,:,f]
            #               augImg[newSeg[s],:,:,f]=img1.copy()             
            # if(False):
            #     #print(' add noise ')
            #     bias = tf.random.uniform([], minval=1.0,maxval=30.0)
            #     for s in range(segments):
            #         for f in range(seqLen):
            #             img1 = CurrentImg[s,:,:,f]
            #             sigma=bias;
            #             n1=sigma*np.random.randn(height,columns)
            #             n2=sigma*np.random.randn(height,columns)
            #             img2 = np.sqrt((img1.copy()+n1)**2+n2**2) 
            #             augImg[s,:,:,f]=img2.copy() 
            
          
            # add augmented data  
            #imgplot = plt.imshow(augImg[:,:,0],cmap='gray')
            #plt.show()
            X.append(augImg.copy())
            orYData=Yor[i,:]
            Y.append(orYData)
            #print('append ',j,len(X))
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    

    
    #X=np.expand_dims(X,axis=5)
    
    return X,Y

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:21.194107Z","iopub.execute_input":"2023-05-29T07:00:21.194537Z","iopub.status.idle":"2023-05-29T07:00:49.839291Z","shell.execute_reply.started":"2023-05-29T07:00:21.194433Z","shell.execute_reply":"2023-05-29T07:00:49.838147Z"}}

# load data from DICOM files
#X, Y = create_data(data_dir)

# load data fom numpy arrays
X=np.load('X_heart_full_raw_global.npy')  # images
Y=np.load('Y_heart_full_raw_global.npy')  # labels
BBraw=np.load('BB_X_heart_crop_raw_global.npy') # LV localization

# load patients codes
patData=dict()
patData= pd.read_csv('PatientsList.csv', encoding= 'latin1') # MIOT Export
patientCodes=patData['PATIENT']
patientClasses=patData['CLASS']


# rearrange localization data in patients, slice, coordinates form
slices=3 # number of slices
BB=np.zeros([X.shape[0],slices,4])
for p in range(X.shape[0]):  #patients
    for s in range(slices):   #slices
        BB[p,s,:]=BBraw[3*p+s,:]
del BBraw

# patietns classes statistics
print('Normal = ',np.sum(Y[:,0]))
print('IronOverload = ',np.sum(Y[:,1]))
print('Severe = ',np.sum(Y[:,2]))

X.shape
Y.shape

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:49.884883Z","iopub.execute_input":"2023-05-29T07:00:49.885395Z","iopub.status.idle":"2023-05-29T07:00:49.891962Z","shell.execute_reply.started":"2023-05-29T07:00:49.885350Z","shell.execute_reply":"2023-05-29T07:00:49.890706Z"}}
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
#for p in range(X.shape[0]):  #patients
#for p in range(3):  #patients
#   for f in range(seq_len):  #patients
#        img=X[p,0+f,:,:,0]  #image
#        plt.subplot(1, 2, 1)
#        imgplot = plt.imshow(img,cmap='gray')
#        plt.subplot(1, 2, 2)
#        img1=X[p,10+f,:,:,0]
#        imgplot = plt.imshow(img1,cmap='gray')
#        plt.show()
#        #time.sleep(0.5)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:49.897308Z","iopub.execute_input":"2023-05-29T07:00:49.897675Z","iopub.status.idle":"2023-05-29T07:00:55.013710Z","shell.execute_reply.started":"2023-05-29T07:00:49.897639Z","shell.execute_reply":"2023-05-29T07:00:55.012662Z"}}

# arrange image data in patietns, slices, rows, columns, frames form
Xvol=np.zeros([X.shape[0],slices,X.shape[2],X.shape[3],seq_len],dtype='uint16')
for p in range(X.shape[0]):  #patients
    for f in range(seq_len): #frames
        for s in range(slices):   #slices
          Xvol[p,s,:,:,f]=X[p,s*seq_len+f,:,:,0]            
print(Xvol.shape)


# crop images and convert in polar coordinates
dimx=Xvol.shape[2] # image dimension
dimy=Xvol.shape[3]

dx=4 # crop enlargement
dy=4 
img=Xvol[0,0,:,:,0]  #sample image
imgCrop=img[int(np.floor(dimx*BB[0,0,0]))-dx:int(np.ceil(dimx*BB[0,0,2]))+dx,int(np.floor(dimy*BB[0,0,1]))-dy:int(np.ceil(dimy*BB[0,0,3]))+dy]

print('crop size  =', imgCrop.shape)
print('int size  =', img.shape,img_height,img_width)

X_crop=np.zeros([Xvol.shape[0],Xvol.shape[1],img_height,img_width,Xvol.shape[4]],dtype='uint16')
for p in range(Xvol.shape[0]):
    for f in range(Xvol.shape[4]):
        for s in range(Xvol.shape[1]):
            img=Xvol[p,s,:,:,f]  #image
            # crop
            imgCrop=img[int(np.floor(dimx*BB[p,s,0]))-dx:int(np.ceil(dimx*BB[p,s,2]))+dx,int(np.floor(dimy*BB[p,s,1]))-dy:int(np.ceil(dimy*BB[p,s,3]))+dy] #crop
            # convert in polar representation
            img1 = imgCrop.astype(np.float32)
            value = int(np.sqrt(((img1.shape[1]/2.0)**2.0)+((img1.shape[0]/2.0)**2.0)))
            polar_image = cv2.linearPolar(img1,(img1.shape[0]/2, img1.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
            polar_image = polar_image.astype(np.uint16)
            X_crop[p,s,:,:,f]=cv2.resize(polar_image,(img_height,img_width))           

X=X_crop
del X_crop


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:00:55.015239Z","iopub.execute_input":"2023-05-29T07:00:55.016010Z","iopub.status.idle":"2023-05-29T07:01:03.617594Z","shell.execute_reply.started":"2023-05-29T07:00:55.015962Z","shell.execute_reply":"2023-05-29T07:01:03.616446Z"}}

# reshape image data set
X.shape
X1=np.zeros([X.shape[0],X.shape[1]*X.shape[2],X.shape[3],X.shape[4]])
for i in range(int(X.shape[0])):
    for s in range (slices):
        for c in range (img_height):    
            for r in range (img_width):
                X1[i,s*img_height+c,r,:]=X[i,s,c,r,:]

# X2=np.zeros([X.shape[0],X.shape[3],X.shape[1]*X.shape[2],X.shape[4]])
# for i in range(int(X.shape[0])):
#     for c in range (X.shape[1]*X.shape[2]):    
#         for r in range (X.shape[3]):
#             X2[i,r,c,:]=X1[i,c,r,:]

seq_len = 5            
X2=np.zeros([X.shape[0],X.shape[3],X.shape[1]*X.shape[2],seq_len])
for i in range(int(X.shape[0])):
    for c in range (X.shape[1]*X.shape[2]):    
        for r in range (X.shape[3]):
            X2[i,r,c,:]=X1[i,c,r,[1,3,5,7,9]]

X2.shape
X1=X2
del X2

dimR=X1.shape[1]
dimC=X1.shape[2]


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:01:03.619211Z","iopub.execute_input":"2023-05-29T07:01:03.619590Z","iopub.status.idle":"2023-05-29T07:01:08.740206Z","shell.execute_reply.started":"2023-05-29T07:01:03.619549Z","shell.execute_reply":"2023-05-29T07:01:08.738333Z"}}
# CNN model

def createFullModel():
    kSize=3
    fSize=10
    bSize=64
    regValue=0.001
    
    #common_input = Input(shape = (dimR, dimC,seq_len),name="common_input")
    
    CNN_Input = Input(shape = (dimR, dimC,seq_len),name="CNN_Input")
    
    CNN_1_1 = Conv2D(filters = bSize, kernel_size = (kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue))(CNN_Input)
    CNN_1_2 = MaxPooling2D(pool_size=(2, 2))(CNN_1_1)
    CNN_1_3 = SpatialDropout2D(0.3)(CNN_1_2)
    
    CNN_2_1 = Conv2D(filters = bSize, kernel_size = (kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue))(CNN_1_3)
    CNN_2_2 = MaxPooling2D(pool_size=(2, 2))(CNN_2_1)
    CNN_2_3 = SpatialDropout2D(0.3)(CNN_2_2)
    
    CNN_3_1 = Conv2D(filters = bSize, kernel_size = (kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue))(CNN_2_3)
    CNN_3_3 = MaxPooling2D(pool_size=(2, 2))(CNN_3_1)
    
    CNN_4_1 = Conv2D(filters = bSize, kernel_size = (kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue))(CNN_3_3)
    CNN_4_3 = MaxPooling2D(pool_size=(2, 2))(CNN_4_1)
    
    CNN_5_1 = Conv2D(filters = bSize, kernel_size = (kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue))(CNN_4_3)
    CNN_5_3 = MaxPooling2D(pool_size=(1, 2))(CNN_5_1)
    
    CNN_6_1 = Flatten(data_format='Channels_last')(CNN_5_3)
    CNN_6_2 = Dropout(0.3)(CNN_6_1)
    CNN_6_3 = Dense(128, activation="relu")(CNN_6_2)
    CNN_6_4 = Dropout(0.3)(CNN_6_3)
    CNN_6_5 = Dense(128, activation="relu")(CNN_6_4)
    
    CNN_Output = Dense(3, activation = "softmax", name = "CNN_Output")(CNN_6_5)
    
    FULL_model = Model(inputs=[CNN_Input],outputs=[CNN_Output], name = "FULL_model")
    return  FULL_model

#FULL_model.summary()
#tf.keras.utils.plot_model(FULL_model, to_file='Full_model.png', show_shapes=True)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-05-29T07:01:08.742550Z","iopub.execute_input":"2023-05-29T07:01:08.743804Z"}}


def step_decay(epoch):
   initial_lrate = 0.0001
   drop = 0.1
   epochs_drop = 20.0
   lrate = initial_lrate * np.power(drop,np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)
cosrate=tf.keras.optimizers.schedules.CosineDecayRestarts(0.00001,50)

opt = tf.keras.optimizers.experimental.AdamW(learning_rate=0.00001,  epsilon=0.01)
opt1=tf.keras.optimizers.SGD(learning_rate=cosrate,momentum=0.9)

#model.compile(loss='categorical_hinge', optimizer=opt, metrics=["accuracy"])
 
earlystop = EarlyStopping(monitor="val_accuracy",patience=200,verbose=1,restore_best_weights=True)
callbacks = [earlystop]

#callbacks = [earlystop,cosrate]




accuracyKFold = []            # global accuracy values (k-fold, last value test set)
ConfusionMatrixKFold = []     # confusion matrices (k-fold, last value test set)

# create test set
ss=StratifiedShuffleSplit(n_splits=2,test_size=0.20,train_size=None,random_state=422)
ss.get_n_splits(X1, Y)
split_1,split_2=ss.split(X1,Y) 

X_train=X1[split_1[0],:,:,:] 
X_test=X1[split_1[1],:,:,:]
y_train=Y[split_1[0],:]
y_test=Y[split_1[1],:]

# associate patients code to test/training set
patientTestCodes=patientCodes[split_1[1]]
patientTestClasses=patientClasses[split_1[1]]
patientTrainCodes=patientCodes[split_1[0]]
patientTrainClasses=patientClasses[split_1[0]]
patientTrainCodesList=patientTrainCodes.tolist()
patientTrainClassesList=patientTrainClasses.tolist()


# k-fold analysis
k=5 # number of folds
y_train1 = np.argmax(y_train, axis = 1)
folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=7 )

for j, (train_idx, val_idx) in enumerate(folds.split(X_train,y_train1)):
    print('\nFold ',j)
    X_val = X_train[val_idx] # validation set for the fold
    y_val = y_train[val_idx]
    X_train2 = X_train[train_idx] # train set for the fold
    y_train2 = y_train[train_idx]
    
    # data augmentation
    X_train_a,y_train_a=createAugmentedDataShift(X_train2,y_train2,6)

    print('Validation set distribution')
    total=np.sum(y_val[:,0])+np.sum(y_val[:,1])+np.sum(y_val[:,2])
    print('Normal = ',np.sum(y_val[:,0]),np.sum(y_val[:,0])/total)
    print('Moderate = ',np.sum(y_val[:,1]),np.sum(y_val[:,2])/total)
    print('Severe = ',np.sum(y_val[:,2]),np.sum(y_val[:,2])/total)

    # set training parameters
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=0.00001,  weight_decay=0.009,epsilon=0.01)
    epochs=5000
    batchSize=8

    # introduce weighted loss
    from sklearn.utils import class_weight
    y_ints = [y.argmax() for y in y_train2]
    classW = class_weight.compute_class_weight('balanced',classes=np.unique(y_ints),y=y_ints)
    classW1 = dict(zip(np.unique(y_ints), classW))
    
    # create model
    FULL_model=createFullModel()
    FULL_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

    # perform training
    history = FULL_model.fit(x = X_train_a, y= y_train_a, epochs=epochs, class_weight=classW1,batch_size = batchSize,shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks)

    #save current model
    CurrentModel='model_'+str(j)
    FULL_model.save(CurrentModel)
    
    # evaluate model performance
    y_pred = FULL_model.predict(X_val,batch_size=batchSize) # predict on validation set

    y_pred1 = np.argmax(y_pred, axis = 1)
    y_val1 = np.argmax(y_val, axis = 1)

    y_pred_label=[None] * np.size(y_pred1)
    for c in range(len(classes)):
        id=np.array(np.where(y_pred1==c))
        for i in range (np.size(id)):
            y_pred_label[id[0,i]]=classes[c]
            
    y_val_label=[None] * np.size(y_val1)
    for c in range(len(classes)):
            id=np.array(np.where(y_val1==c))
            for i in range (np.size(id)):
                y_val_label[id[0,i]]=classes[c]

    print(classification_report(y_val_label, y_pred_label,labels=classes))
    print('Accuracy = ',accuracy_score(y_val_label,y_pred_label))
    classesCM = ["Norm","Moderate","Severe"]
    confusionMatrix=confusion_matrix(y_val_label,y_pred_label,labels=classes,normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classesCM)
    disp.plot()
        
    accuracyKFold.append(accuracy_score(y_val_label,y_pred_label))
    ConfusionMatrixKFold.append(confusionMatrix)

    # save performence data 
    data = []
    data.append(['patientsCode','patientsClass','y_test_label','y_pred_label']) 

    for f in range(len(y_val_label)):
        data.append([patientTrainCodesList[val_idx[f]],patientTrainClassesList[val_idx[f]],y_val_label[f],y_pred_label[f]]) 
    df = pd.DataFrame(data)
    file = 'ValSetResults'+str(j)+'.csv'
    df.to_csv(file) 
        

#find and load the best model
idm=np.argmax(accuracyKFold)
BestModel='model_'+str(idm)
best_model = tf.keras.models.load_model(BestModel)

# evaluete performences on the test set
y_pred = best_model.predict(X_test, batch_size=batchSize)  # predit on test set

y_pred1 = np.argmax(y_pred, axis=1)
y_test1 = np.argmax(y_test, axis=1)

y_pred_label = [None] * np.size(y_pred1)
for c in range(len(classes)):
    id = np.array(np.where(y_pred1 == c))
    for i in range(np.size(id)):
        y_pred_label[id[0, i]] = classes[c]

y_test_label = [None] * np.size(y_test1)
for c in range(len(classes)):
    id = np.array(np.where(y_test1 == c))
    for i in range(np.size(id)):
        y_test_label[id[0, i]] = classes[c]


data = []
patientTestClassesList=patientTestClasses.tolist()
patientTestCodesList=patientTestCodes.tolist()
data.append(['patientsCode','patientsClass','y_test_label','y_pred_label']) 

# save performence data
for f in range(len(patientTestCodes)):
    data.append([patientTestCodesList[f],patientTestClassesList[f],y_test_label[f],y_pred_label[f]]) 
df = pd.DataFrame(data)
file = 'TestSetResults.csv'
df.to_csv(file) 

print(classification_report(y_test_label, y_pred_label,labels=classes))
print('Accuracy = ',accuracy_score(y_test_label,y_pred_label))
confusionMatrix=confusion_matrix(y_test_label,y_pred_label,labels=classes,normalize=None)
classesCM = ["Norm","Moderate","Severe"]
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classesCM)
disp.plot()
    
accuracyKFold.append(accuracy_score(y_test_label,y_pred_label))
ConfusionMatrixKFold.append(confusionMatrix)    


# save on file
import pickle
with open('kfoldResultsAccuracy.pkl', 'wb') as f:
    pickle.dump(accuracyKFold, f)
with open('kfoldResultsCM.pkl', 'wb') as f:    
    pickle.dump(ConfusionMatrixKFold, f)

with (open('kfoldResultsAccuracy.pkl', "rb")) as openfile:
    acc=pickle.load(openfile)



# %% [code] {"jupyter":{"outputs_hidden":false}}


meanAcc=np.mean(accuracyKFold[0:4])
sdAcc=np.std(accuracyKFold[0:4])

print('val accuracy = ',meanAcc,' ',sdAcc)
print('test accuracy = ',accuracyKFold[5])



print('Normal = ',np.sum(y_val[:,0]))
print('IronOverload = ',np.sum(y_val[:,1]))
print('Severe = ',np.sum(y_val[:,2]))


# # %% [code] {"jupyter":{"outputs_hidden":false}}
# loss =  history.history['loss']
# accuracy = history.history['accuracy']

# val_loss =  history.history['val_loss']
# val_accuracy = history.history['val_accuracy']

# epochs_range = range(len(history.history['loss']))

# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, accuracy, label='Trainig Accuracy')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
# plt.legend(loc='upper right')
# plt.title('Training Loss and Accuracy')
# plt.ylim([0, 1])
# plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
    
