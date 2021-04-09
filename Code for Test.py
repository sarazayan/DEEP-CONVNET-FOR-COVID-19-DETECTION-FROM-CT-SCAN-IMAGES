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
           
  


