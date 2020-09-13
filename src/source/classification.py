

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
'''
The following dummy code for demonstration.
'''

def image_from_array(arr,width,height,return_image=False):
    """
    Input : Takes in an array, width and height
    Output: Displays an image by reshaping image to 
            provided width and height.
    
    
    # Press any key to close the window 
    # if return_image=True, then the image matrix is returned
    # instead of displaying it.
    """
    # Reshaping the given array
    img = np.array(arr.reshape(width,height),dtype=np.uint8)
    
    if return_image:
        return img
    
    else:
        # displaying image ; press any button to close
        cv.imshow('image',img)
        cv.waitKey(0)
        cv.destroyAllWindows()

def generate_images(data):
    """
    Input : Input data matrix of images
    Output: Store all the images in the /test_images/ folder
    """
   
    # Getting current working directory
    path = os.getcwd()
    
    try:  
        # Creating a new directory 'test_images'
        os.mkdir(path+'/test_images')  
        
    except OSError as error:  
        # If directory already exists
        print(error)
        print("\nDelete the existing test_images folder & try again")
    store_path = path+'/test_images/'
    for i in range(len(data)):
        img = image_from_array(data[i],48,48,return_image=True)
        cv.imwrite(store_path+str(i)+'.png',img)

def generate_images_folderwise(x_train,y_train,x_test,y_test):
    """
    Input : Input data matrix of images, lables list 
    Output: Store all the images in the 
            /images/<train or test>/<class_folder> 
    """
    # Checking if given data matrix and labels match
    assert len(x_train)==len(y_train),"Input array size ≠ labels size"
    assert len(x_test)==len(y_test),"Input array size ≠ labels size"
    
    # Getting current working directory
    path = os.getcwd()
    
    temp = 0
    
    try:  
        # Creating a new directory 'images'
        os.mkdir(path+'/images')
        os.mkdir(path+'/images/'+'train')
        os.mkdir(path+'/images/'+'test')
        
    except OSError as error:  
        # If directory already exists
        print(error)
        print("\nDelete the existing images folder & try again")
        
    store_path = path+'/images/'+'train/'
    class_list = np.unique(y_train)
    for j in class_list:
        temp = 0
        os.mkdir(store_path+str(j))
        for i in range(len(x_train)):
            if y_train[i]==j:
                temp += 1
                img = image_from_array(x_train[i],48,48,return_image=True)
                cv.imwrite(store_path+str(j+'/')+str(temp)+'.png',img)
                
    store_path = path+'/images/'+'test/'
    class_list = np.unique(y_test)
    for j in class_list:
        temp = 0
        os.mkdir(store_path+str(j))
        for i in range(len(x_test)):
            if y_test[i]==j:
                temp += 1
                img = image_from_array(x_test[i],48,48,return_image=True)
                cv.imwrite(store_path+str(j+'/')+str(temp)+'.png',img)


def train_a_model(trainfile):
    '''
    :param trainfile:
    :return:
    '''
    x = trainfile.iloc[:,1:].values
    y = trainfile['emotion'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y,random_state=42)
    generate_images_folderwise(x_train,y_train,x_test,y_test)
    path = os.getcwd()

    TRAINING_DIR = path+"/images/train/"
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = path+"/images/test/"
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        color_mode='grayscale',
        target_size = (48,48),
        batch_size = 64,
        class_mode='categorical',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        color_mode = 'grayscale',
        target_size = (48,48),
        batch_size = 64,
        class_mode='categorical',
        shuffle=False)

    print(train_generator.class_indices)


    num_classes = 3
    epochs = 100

    # Creating the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),strides=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(48,48,1),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 1),strides=2))
    model.add(Dropout(0.25))



    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
    loss = 'categorical_crossentropy', 
    optimizer=Adam(), 
    metrics=['accuracy'])

    steps_per_epoch = train_generator.n//train_generator.batch_size
    validation_steps = validation_generator.n//validation_generator.batch_size

    history = model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps = validation_steps,
    )

    model.save('model')

    pass



def test_the_model(testfile):
    '''

    :param testfile:
    :return:  a list of predicted values in same order of
    '''
    path = os.getcwd()
    m = tf.keras.models.load_model('model')
    output = []
    generate_images(testfile.values)
    temp = 0
    for i in range(len(testfile)):
        img = cv.imread(path+"/test_images/"+str(i)+".png", cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(48,48))
        img = img.reshape(1, 48, 48, 1)
        img = img/255.0
        temp = m.predict(img)
        temp = np.argmax(temp, axis = 1)
        if temp ==0:
            output.append('Fear')
        elif temp==1:
            output.append('Happy')
        elif temp==2:
            output.append('Sad')
    print(output)
    return output
