import numpy as np
import struct as st
import sys 
from sklearn.decomposition import PCA


  
def readMnistData(N,filename):
     imgs = open(filename+'/'+'train-images.idx3-ubyte','rb')
     labels = open(filename+'/'+'train-labels.idx1-ubyte','rb')

     image_size = 28
     num_images = 800


     imgs.read(16)
     buf = imgs.read(image_size * image_size * num_images)
     data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)  
     data = data.reshape(num_images, image_size * image_size)

     labels.read(8)
     labelsbuf = labels.read(num_images)
     labelsdata = np.frombuffer(labelsbuf, dtype=np.uint8).astype(np.float32)  
     labelsdata = labelsdata.reshape(num_images)
     
     testLables, trainLables=labelsdata[:N], labelsdata[N:]
     testImages, trainImages=data[:N], data[N:]

     print (np.mean(testImages[0]))
     print (testLables.shape)
     print (trainLables.shape)
     print (testImages.shape)
     print (trainImages.shape)

     return  testLables, trainLables, testImages, trainImages
#############################################
#Load Images from the Minst Dataset
#############################################
def read_idx(filename):
    with open(filename) as f:
        zero, data_type, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)



#############################################
#Calculate Eucledian Distance          
#############################################
def euclideanDistance(X, Y):
    return np.linalg.norm(X - Y);



#############################################
#Reshape and Split test and train set
#############################################    
def load_DataSet(N,filepath):
   
   images = read_idx(filepath+'/'+'train-images-idx3-ubyte')
   labels = read_idx(filepath+'/'+'train-labels-idx1-ubyte')
  
   reshaped_images = np.reshape(images, (60000, 28 * 28))
   reshaped_images=reshaped_images[:800]

   reshaped_labels =labels
   reshaped_labels=reshaped_labels[:800]

   testLables, trainLables=reshaped_labels[:N], reshaped_labels[N:]
   testImages, trainImages=reshaped_images[:N], reshaped_images[N:]

   print (np.mean(testImages[0]))
   print (testLables.shape)
   print (trainLables.shape)
   print (testImages.shape)
   print (trainImages.shape)

   return  testLables, trainLables, testImages, trainImages




#################################################################
#Apply PCA to transform test set and transform and fit train set
#################################################################
def my_PCA(D,testImages,trainImages):   
   
   reducedTestImages= np.zeros([])
   reducedTrainImages= np.zeros([])
   pca = PCA(n_components=D,svd_solver='full')
   
   reducedTrainImages=pca.fit_transform(trainImages)
   reducedTestImages=pca.transform(testImages)
   print (reducedTestImages[0])
   
   return reducedTestImages, reducedTrainImages





#################################################################
#Get K-Nearest Neighbour for the given testimage
#################################################################
def get_K_nearest_neighbors(trainimages, testimage,k):
   distances = []
   for j in range(len(trainimages)):
      dist = euclideanDistance(testimage, trainimages[j])
      distances.append([j, dist])

   distances.sort(key=lambda tup: tup[1])
   neighbors = []
   for i in range(k):
      neighbors.append(distances[i])
   #print (neighbors)
   return neighbors


#################################################################
#Predict Label for given test image
#################################################################
def predict_class_for_test_image(testimage,trainimages,trainlabels,k):
   predicted_label=np.zeros(10)
   neighbors = get_K_nearest_neighbors(trainimages, testimage, k)
   # print (neighbors.shape)
   #print (trainlabels.shape)
   for i in range(len(neighbors)):
      index=int(trainlabels[neighbors[i][0]])
      #print index
      predicted_label[index] = predicted_label[index] + (1 / neighbors[i][1]) 
   prediction = np.argmax(predicted_label)
   print (prediction)
   return prediction 


#################################################################
#Predict Labels for given test set
#################################################################
def predict_classes_for_test_set(testimages,testlabels,trainimages,trainlabels,k):
   f = open("Output.txt","w+")
   #print (trainlabels.shape)
   acc=0
   # print (testlabels)
   for i in range(len(testimages)):
      predicted_label=predict_class_for_test_image(testimages[i],trainimages,trainlabels,k)
      #print (predicted_label)
      #print (testlabels[i])
      print(predicted_label,int(testlabels[i]))      
      f.write("{0} {1}\n".format(str(predicted_label) , str(int(testlabels[i])) ))

   f.close()
   

K= int(sys.argv[1])
D=int(sys.argv[2])
N=int(sys.argv[3])
filepath=sys.argv[4]

print (K,D,N,filepath)
testLables, trainLables, testImages, trainImages = readMnistData(N,filepath)

my_PCA(D,testImages,trainImages)
predict_classes_for_test_set(testImages,testLables,trainImages,trainLables,N)
#readMnistData(20,'./mnist')
