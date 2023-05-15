"""
References:
[1] https://github.com/MarkPrecursor/SRCNN-keras
[2] https://github.com/kunal-visoulia/Image-Restoration-using-SRCNN
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from skimage.measure import compare_ssim as ssim
import cv2
import numpy as np
import math 
import os

def psnr(target,ref):#target image and refernce image
    
    #assume RGB image and convert all integer values to float
    target_data=target.astype(float)
    ref_data=ref.astype(float)
    
    diff=ref_data-target_data
    diff=diff.flatten('C')#need ot flatten so computations can be done
    
    rmse=math.sqrt(np.mean(diff**2.))#2. for float values
    
    return 20*math.log10(255./rmse)

def mse(target,ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err=np.sum((target.astype('float')-ref.astype('float'))**2)
    err=err/float(target.shape[0]*target.shape[1])#divided by total number of pixels
    
    return err

def compare_images(target,ref):
    scores=[]
    scores.append(psnr(target,ref))
    scores.append(mse(target,ref))
    scores.append(ssim(target,ref,multichannel=True))#multichannel so that it can handle 3Dor 3 channel images RGB/BGR 
    
    return scores


def make_model():
    SRCNN = Sequential(
        [
            Input((None, None, 3)),
            Conv2D(filters=128, kernel_size=(9,9), activation="relu", padding="valid"),
            Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
            Conv2D(filters=1, kernel_size=(5,5), activation="linear", padding="valid"),
        ]
    )
    return SRCNN

def prepare_images(path, factor):
    
    # loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        # find old and new image dimensions
        h, w, c = img.shape
        new_height = int(h / factor)
        new_width = int(w / factor)
        
        # resize the image - down
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR) #interploation are methods for resizing images;how do you go from image with 100px to 1000px 
        #bilinear interpolation
        
        # resize the image - up
        img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite('images/{}'.format(file), img)

def main():
    srcnn_model = make_model()
    srcnn_model.summary()
    
    adam = Adam(learning_rate=0.0003) #different from paper
    srcnn_model.compile(optimizer=adam, loss="mean_squared_error", metrics=['mean_squared_error'])
    
    # import training and test data
    prepare_images('source/',2)
    #source folder has high resolution images that will be converted to low resoltion images to be used for SRCNN

if __name__ == "__main__":
    main()