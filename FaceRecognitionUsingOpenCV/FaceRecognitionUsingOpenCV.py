
# coding: utf-8

# Face Recognition with OpenCV

from array import array
from multiprocessing.dummy import Array
#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as 
#it is needed by OpenCV face recognizers
import numpy as np

#chdir in order to adjust path changes
os.chdir('FaceRecognitionUsingOpenCV/')
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Ferdinan Kurnianto", "Tikkos"]


#function to detect face using OpenCV
def detect_face_lbp(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using haar in this one
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#function to detect face using OpenCV
def detect_profile_lbp(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using haar in this one
    profile_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_profileface.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of profiles
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.2);
    
    #if no profiles are detected then return original img
    if (len(profiles) == 0):
        return None, None
    
    #under the assumption that there will be only one profile,
    #extract the profile area
    (x, y, w, h) = profiles[0]
    
    #return only the profile part of the image
    return gray[y:y+w, x:x+h], profiles[0]


#function to detect face using OpenCV
def detect_face_haar(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = []
    #load OpenCV face detector, I am using haar in this one
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_default.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #extract the face area
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        rect.append(gray[y:y+w, x:x+h])
    
    #return only the face part of the image
    return rect, faces

#function to detect face using OpenCV
def detect_face_haar2(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using haar in this one
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt2.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#function to detect face using OpenCV
def detect_profile_haar(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = []
    #load OpenCV face detector, I am using haar in this one
    profile_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_profileface.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of profiles
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.2);
    
    #if no profiles are detected then return original img
    if (len(profiles) == 0):
        return None, None
    
    #under the assumption that there will be only one profile,
    #extract the profile area
    (x, y, w, h) = profiles[0]
    rect.append(gray[y:y+w, x:x+h])
    #return only the profile part of the image
    return rect, profiles[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
    
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
            
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        
        #build path of directory containin images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
            
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #display an image window to show the image 
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            #detect face
            face, rect = detect_face_haar(image)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face[0])
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict_lbp(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_lbp(img)

    #predict the image using our face recognizer 
    try:
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        print("Prediction failed")
    return img

def predict_lbp2(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_lbp(img)
    if face is None:
        face, rect = detect_profile_lbp(img)

    #predict the image using our face recognizer 
    try:
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        print("Prediction failed")
    return img

def predict_haar(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_haar(img)

    #predict the image using our face recognizer 
    try:
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        print("Prediction failed")
    return img


def predict_haar2(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_haar(img)
    if face is None:
        face, rect = detect_profile_haar(img)

    #predict the image using our face recognizer 
    try:
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        print("Prediction failed")
    return img

def predict_haar3(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_haar2(img)
    if face is None:
        face, rect = detect_profile_haar(img)

    #predict the image using our face recognizer 
    try:
        label, confidence = face_recognizer.predict(face)
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
    except:
        print("Prediction failed")
    return img

def predict_haar4(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    imgresult = []
    #detect face from the image
    face, rect = detect_face_haar(img)
    if face is None:
        face, rect = detect_profile_haar(img)
    if face is None:
        img = cv2.flip(img,1)
        face, rect = detect_profile_haar(img)
    #predict the image using our face recognizer
    try:
        if(len(face)==1 and type(rect[0]) is not np.ndarray):
            label, confidence = face_recognizer.predict(face[0])
            #get name of respective label returned by face recognizer
            label_text = subjects[label]
                  
            #draw a rectangle around face detected
            draw_rectangle(img, rect)
            #draw name of predicted person
            draw_text(img, label_text, rect[0], rect[1]-5)
            (x, y, w, h) = rect
            img = img[0:y+w+150, x-100:x+h+150]
            imgresult.append(img)
    except:
        print("Prediction failed")
        imgresult.append(img)
    try:
        if(type(rect[0]) is np.ndarray):
            for i in range(0, len(face)):
                label, confidence = face_recognizer.predict(face[i])
                #get name of respective label returned by face recognizer
                label_text = subjects[label]
                        
                #draw a rectangle around face detected
                draw_rectangle(img, rect[i])
                #draw name of predicted person
                draw_text(img, label_text, rect[i][0], rect[i][1]-5)
                (x, y, w, h) = rect[i]
                img2 = img[y-70:y+w+150, x-100:x+h+150]
                imgresult.append(img2)
    except:
        print("Prediction failed")
    return imgresult



if(os.path.exists('face-recognizer/recognizer.xml')==False):
    #let's first prepare our training data if we don't have a saved trained model
    #data will be in two lists of same size
    #one list will contain all the faces
    #and other list will contain respective labels for each face
    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))
    #save our trained face recognizer
    face_recognizer.write('face-recognizer/recognizer.xml')
else:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    #load our trained face recognizer
    face_recognizer.read('face-recognizer/recognizer.xml')

test_img=[0]*22
predicted_img=[0]*22

print("Predicting images...")

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_lbp(test_img[x])
    print("Prediction complete")

    #display both images
    cv2.imwrite('output/lbp/Output'+str(x)+'.png', predicted_img[x])

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_lbp2(test_img[x])
    print("Prediction complete")

    #display both images
    cv2.imwrite('output/lbp+profile/Output'+str(x)+'.png', predicted_img[x])

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_haar(test_img[x])
    print("Prediction complete")

    #display both images
    cv2.imwrite('output/haar/Output'+str(x)+'.png', predicted_img[x])

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_haar2(test_img[x])
    print("Prediction complete")

    #display both images
    cv2.imwrite('output/haar+profile/Output'+str(x)+'.png', predicted_img[x])

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_haar3(test_img[x])
    print("Prediction complete")

    #display both images
    cv2.imwrite('output/haar2+profile/Output'+str(x)+'.png', predicted_img[x])

for x in range(1, len(test_img)):
    #load test images
    test_img[x] = cv2.imread("test-data/test"+str(x)+".jpg")

    #perform a prediction
    predicted_img[x] = predict_haar4(test_img[x])
    print("Prediction complete")
    #if more than 1 face is predicted, save each face as a different file
    if(len(predicted_img[x])>1):
        for i in range(0, len(predicted_img[x])):
            cv2.imwrite('output/haar+profile+flip/Output'+str(x)+'-'+str(i)+'.png', predicted_img[x][i])
    #display both images
    else:
        cv2.imwrite('output/haar+profile+flip/Output'+str(x)+'.png', predicted_img[x][0])