import cv2
from matplotlib import pyplot as plt
import numpy as np
import os


def whatError(h1, h2):
    err = 0
    #calculating the residual error
    for i in range(256):
        err = err + (abs(h1[i]-h2[i]))**2
    return err

def which_class(mean_feature_vector, image_hist):
    error = []
    for i in range(len(mean_feature_vector)):
        error.append(whatError(mean_feature_vector[i], image_hist))
    return error.index(min(error))

def add_vectors(v1, v2):
    v3 = []
    for i in range(len(v1)):
        v3.append(v1[i] + v2[i]) 
    return v3

def scalar_vector_division(v1, k):
    for i in range(len(v1)):
        v1[i]/=k
    return v1
   
curDir = "./Texture training set"
print(curDir)
#list files
list_of_files = os.listdir(curDir)


  
img = list()
feature_vector = []
#preparing the image list
i=0
for image_name in list_of_files:
    image = cv2.imread("./Texture training set\\"+image_name,0)
    img.append(image)
    
for image in img:
    hist = cv2.calcHist(image, [0],None, [256], [0, 256])
    feature_vector.append(hist)


j=0

no_of_variants_of_each_class = 3 #same for each class
no_of_classes = 5
mean_feature_vector = [[0 for _ in range(256)]]*no_of_classes

for i in range(no_of_classes):
    for k in range(no_of_variants_of_each_class):
        mean_feature_vector[i] = add_vectors(mean_feature_vector[i],feature_vector[j])
        j+=1

for i in range(no_of_classes):
    mean_feature_vector[i] = scalar_vector_division(mean_feature_vector[i],no_of_variants_of_each_class)

"""
new_image_name = input("enter image name with path: ")
new_image = cv2.imread(new_image_name, 0)
cv2.imshow(new_image_name, new_image)
image_hist = cv2.calcHist(new_image, [0], None, [256], [0, 256])
print("the new image is labeled to class ",which_class(mean_feature_vector, image_hist))
"""
## calculating the confusion matrix
no_of_classes = 5
no_of_variants_of_each_class = 5
mat = [[0 for _ in range(no_of_variants_of_each_class)] for _ in range(no_of_classes)]
i=1
h = 0
v = 0
path_to_dataset = "./whole data set"
#path_to_dataset = input("Please provide the path to the dataset:")
list_of_files = os.listdir(path_to_dataset)
for image_name in list_of_files:
    image = cv2.imread(path_to_dataset+"\\"+image_name,0)
    image_hist = cv2.calcHist(image, [0], None, [256], [0, 256])
    if(h==which_class(mean_feature_vector, image_hist)):
        mat[h][h]= mat[h][h] + 1
    else:
        mat[h][v] = mat[h][v] + 1
    if(i%5==0):
        h+=1
        v=-1

    v+=1
    i+=1

##diagonal sum
c=0
diagonal_sum = 0
matrix_sum = 0
print("The Confusion matrix is:")
for row in mat:
    print(row)
    
for i in range(no_of_classes):
    for j in range(no_of_variants_of_each_class):
        if(i==j):
            diagonal_sum = diagonal_sum+mat[i][j]
        else:
            matrix_sum = matrix_sum+mat[i][j]

lst = []
i=0
for row in mat:
    lst.append(row[i]/sum(row))
    i+=1
precision = (sum(lst)/len(lst))*100
print("Accuracy: ",(diagonal_sum/(matrix_sum+diagonal_sum))*100)
print("Precision: ",precision)













            





    
