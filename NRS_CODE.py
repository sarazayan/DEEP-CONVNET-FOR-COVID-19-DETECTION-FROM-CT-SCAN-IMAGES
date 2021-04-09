from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow-io
  import matplotlib.pyplot as plt
  import numpy as np
  from PIL import Image  
  import PIL  
  import numpy as np
  import io
  import os
  import imageio
  import time
 

  import tensorflow as tf
  import tensorflow_io as tfio

  c=['COVID-19 subjects/','Normal subjects/','Cap subjects/' ]
  
   
  for i in range(len(c)): 
    src=(os.path.join( "/content/drive/MyDrive/Dataset/", c[i])) #dataset is the path of the original dicom files 
    #src = "/content/drive/MyDrive/"+c[i]
    dst = os.path.join("/content/drive/MyDrive/Covid_Data", c[i])#destination path of empty folders with same configuration of folders in the original dataset,we used copywhiz to create empty folders with same configuration fo the original dataset

  
    
    
    for root, dirs, files in os.walk(src):
      
      for file in files: 
        

        if ".dcm" in file:# exclude non-dicoms, good for messy folders
           
        
           
           image_bytes = tf.io.read_file(os.path.join(root, file))
           image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

           skipped = tfio.image.decode_dicom_image(image_bytes, on_error='skip', dtype=tf.uint8)

           lossy_image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)
           I=np.squeeze(lossy_image.numpy())
           
           img = Image.fromarray(I)
           os.path.join(dst,root[root.rfind("/")+1:],file[:-4]+'.jpeg')
           new_directory=
           print(new_directory)
           
           

          
           

           img.save(new_directory)





!pip install tensorflow==2.2.0
!pip install Keras==2.2.0
!pip install tensorflow==1.14
!pip install Keras==2.0.8
!pip install Keras-Applications
!pip install keras_applications==1.0.4
!pip install tensorflow
!pip install keras==2.3.1

!pip install -U -q PyDrive
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "1"
config.allow_soft_placement=True
config.log_device_placement=True
config.gpu_options.allocator_type = 'BFC'
set_session(tf.Session(config=config))

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger



from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16

import keras.metrics

from keras.optimizers import Adam, RMSprop

import numpy as np





from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve



from sklearn.metrics import auc





#import matplotlib.pyplot as plt





import numpy, scipy.io

img_height=224

img_width=224



# create a data generator

datagen = ImageDataGenerator()

batch_size=64

# load and iterate training dataset

train_it = datagen.flow_from_directory('/content/drive/MyDrive/Dataset_slice/Train',target_size=(img_height, img_width),class_mode='categorical', batch_size=batch_size)

val_it = datagen.flow_from_directory('/content/drive/MyDrive/Dataset_slice/Valid',target_size=(img_height, img_width),class_mode='categorical', batch_size=batch_size)





# load and iterate test dataset

test_it = datagen.flow_from_directory('/content/drive/MyDrive/Test.jpeg/Test1',target_size=(img_height, img_width),class_mode='categorical', batch_size=batch_size)


# confirm the iterator works

batchX, batchy = train_it.next()

print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


from keras.layers import Flatten

def model(input_img):
    conv1 = Conv2D(5, (3, 3), padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    relu1 = Activation('relu')(pool1)
    drop1 = Dropout(rate = 0.5)(relu1)
    conv2 = Conv2D(5, (3, 3), padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64

    # Added this layer to flatten the input to Dense layer
    flattened = Flatten()(pool2)

    relu2 = Activation('relu')(flattened)
    drop2 = Dropout(rate=0.5)(relu2)
    dense = Dense(2, activation='softmax')(drop2) # 28 x 28 x 1
    return dense

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    print("focal loss done")
    
    return focal_loss


#Declare model

vgg16 = VGG16(weights=None, include_top=True)
#Add a layer where input is the output of the  second last layer 
x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)
model = Model(input=vgg16.input, output=x)
model.summary()


metrics = ['accuracy']
optimizer = Adam(lr=1e-5, decay=1e-6)
steps_per_epoch = (8238) // batch_size

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=2)
    
    
model.compile(loss=categorical_focal_loss(), optimizer=optimizer,metrics=metrics)


# Train the model
model.fit_generator(train_it, 
epochs=20,

steps_per_epoch=steps_per_epoch,
validation_data=test_it,
callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)],
verbose=1, 
workers=1,
validation_steps=40)

#extract features


def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print ("Model loaded..!")
  print (base_model.summary())
  return base_model

def extract_features_and_store(train_generator,validation_generator,base_model):
  
  x_generator = None
  y_lable = None
  batch = 0
  for x,y in train_generator:
      if batch == (int(31268/batch_size)): #31268 is len(train_it)
          break
      print ("predict on batch:",batch)
      print((31268/batch_size))
      

      if batch==0:
        #  if np.array([]).size: print(1)
         x_generator = base_model.predict_on_batch(x)
         y_lable = y
        #  print (y)
         print('x')
      else:
        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
        y_lable = np.append(y_lable,y,axis=0)
        print('g')
      batch+=1
  
  print('s')
  # x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save('train_x.npy',x_generator) 
  np.save('train_y.npy',y_lable)
  batch = 0
  x_generator = None
  y_lable = None
  for x,y in validation_generator:
      if batch == int(14696/batch_size): #15696 is len(val_it)
          
          break
      print ("predict on batch validate:",batch)
      
      if batch==0:
         x_generator = base_model.predict_on_batch(x)
         y_lable = y
      else:
        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
        y_lable = np.append(y_lable,y,axis=0)
      batch+=1
  # x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save('Vali_x.npy',x_generator)
  np.save('Vali_y.npy',y_lable)
  return None
model=load_VGG16_model()
extract_features_and_store(train_it ,val_it,model)
#test
# load the image
# !pip install Pillow imageio
# !pip install scipy==1.1.0
# !pip install scipy==1.2
# !pip install scipy
# !pip install imread
# !pip install imresize
# !pip install imsave
# from PIL import Image
# from skimage.transform import resize
# from skimage import data


# import imageio
import cv2
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import numpy as np
import glob,os
# from scipy.misc import imread,imresize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
#import utils
import os
%matplotlib inline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print("Tensorflow version:", tf.__version__)
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# model = VGG16(weights='None', include_top=False)
import numpy, scipy.io
vgg16 = VGG16(weights=None, include_top=True)
#Add a layer where input is the output of the  second last layer 
x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)
model = Model(vgg16.input, x)
model.summary()
model.load_weights('/content/drive/MyDrive/weights.0003-0.13.hdf5',by_name=False,skip_mismatch=False, options=None)
# model=load_model('my_model.h5')
# model = VGG16(weights='/content/drive/MyDrive/weights.0003-0.13.hdf5', include_top=False)
# predictions_2 = model.predict(validation_generator,batch_size=64,verbose=2)
# print(predictions_2)



# from imageio import imread,imsave

# from scipy.misc import imread,imresize



Test=['Test1']

x = []
y = []
count = 0
output = 0
count_video = 0
correct_video = 0
total_video = 0
print(Test)
image_path=[]
predi=[]

for test_class in Test:


      
      test_class =os.path.join("/content/drive/MyDrive/‘ICASSP SPGC2021 Test" , test_class) #test directory 
      print(test_class)
      file=os.listdir(test_class)
      print(file)

      for patient_i in file:
          patient =os.path.join(test_class, patient_i)
          file_2=os.listdir(patient)
          print(file_2)
          cov=0
          cap=0


          for f in sorted(file_2): 

            f_1=os.path.join(patient,f)
            print(f_1)
            

            image = cv2.imread(f_1)
            
            # orig = image.copy()
          # # pre-process the image for classification
            image = cv2.resize(image, (224, 224))
            # image = image.astype("float") / 255.0
            # image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            print(image.shape)
            pred = model.predict(image)
            pred = labels[np.argmax(pred)]
            if pred=='Cap':
              cap+=1
            elif pred=='Covid':
              cov+=1
          image_path.append(patient_i)
          if cap>1 :
            predi.append('CAP')
          elif cov>=len(file_2)/2:


            print(pred)
          
            predi.append('COVID_19')
          else :
            predi.append('Normal')

 
            import pandas as pd
            d= dict(zip(image_path, predi))
            df = pd.DataFrame.from_dict(d, orient="index")
            df.to_csv("Test_1.csv")
           
  

Test=['Test2']

x = []
y = []
count = 0
output = 0
count_video = 0
correct_video = 0
total_video = 0
print(Test)
image_path=[]
predi=[]

for test_class in Test:


      
      test_class =os.path.join("/content/drive/MyDrive/‘ICASSP SPGC2021 Test" , test_class)
      print(test_class)
      file=os.listdir(test_class)
      print(file)

      for patient_i in file:
          patient =os.path.join(test_class, patient_i)
          file_2=os.listdir(patient)
          print(file_2)
          cov=0
          cap=0


          for f in sorted(file_2): 

            f_1=os.path.join(patient,f)
            print(f_1)
            

            image = cv2.imread(f_1)
            
            # orig = image.copy()
          # # pre-process the image for classification
            image = cv2.resize(image, (224, 224))
            # image = image.astype("float") / 255.0
            # image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            print(image.shape)
            pred = model.predict(image)
            pred = labels[np.argmax(pred)]
            if pred=='Cap':
              cap+=1
            elif pred=='Covid':
              cov+=1
          image_path.append(patient_i)
          if cap>1 :
            predi.append('CAP')
          elif cov>=len(file_2)/2:


            print(pred)
          
            predi.append('COVID_19')
          else :
            predi.append('Normal')

 
            import pandas as pd
            d= dict(zip(image_path, predi))
            df = pd.DataFrame.from_dict(d, orient="index")
            df.to_csv("Test_2.csv")
           
  
Test=['Test3']

x = []
y = []
count = 0
output = 0
count_video = 0
correct_video = 0
total_video = 0
print(Test)
image_path=[]
predi=[]

for test_class in Test:


      
      test_class =os.path.join("/content/drive/MyDrive/‘ICASSP SPGC2021 Test" , test_class)
      print(test_class)
      file=os.listdir(test_class)
      print(file)

      for patient_i in file:
          patient =os.path.join(test_class, patient_i)
          file_2=os.listdir(patient)
          print(file_2)
          cov=0
          cap=0


          for f in sorted(file_2): 

            f_1=os.path.join(patient,f)
            print(f_1)
            

            image = cv2.imread(f_1)
            
            # orig = image.copy()
          # # pre-process the image for classification
            image = cv2.resize(image, (224, 224))
            # image = image.astype("float") / 255.0
            # image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            print(image.shape)
            pred = model.predict(image)
            pred = labels[np.argmax(pred)]
            if pred=='Cap':
              cap+=1
            elif pred=='Covid':
              cov+=1
          image_path.append(patient_i)
          if cap>1 :
            predi.append('CAP')
          elif cov>=len(file_2)/2:


            print(pred)
          
            predi.append('COVID_19')
          else :
            predi.append('Normal')

 
            import pandas as pd
            d= dict(zip(image_path, predi))
            df = pd.DataFrame.from_dict(d, orient="index")
            df.to_csv("Test_3.csv")
           
  


