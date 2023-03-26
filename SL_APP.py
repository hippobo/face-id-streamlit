import streamlit as st 
import PIL.Image

from bson.binary import Binary

import glob
import torchvision.transforms as transforms
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn

import pickle
import cv2
import torch
import os
import shutil
import time
import numpy as np

from pymongo import MongoClient, errors

if torch.cuda.is_available():
    DEVICE = 'cuda'
else :
     DEVICE = 'cpu'

MODELPATH = "trained_model2.pth"
MODEL2 = "trained_model.pth"

##comment code below to run on cpu  

class SiameseNet(nn.Module):


    def __init__(self):
        super(SiameseNet,self).__init__()
       
       
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            
        )


        
        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            
            nn.Linear(512,256),
            nn.ReLU(),

            nn.Linear(256,128),
         
        )
        
    def forward(self, x):
        
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output







def preprocess(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors = 7, minSize = (100,100), flags = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in face:
        face = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        embedding = create_verif_embedding(userCreation = False,userID = None,frame = face)
        
        
        return embedding
    
@st.cache_resource
def load_model(modelPath):
    model = SiameseNet().to(DEVICE)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    
    return model


def create_verif_embedding(userCreation,userID = None,frame = None,modelPath=MODELPATH):
        
        images_list = []
        if userCreation:
            
            FolderPath = "app_data/user_images/" + userID
            for picture in glob.iglob(f'{FolderPath}/*'):
                images_list.append(picture)
            

            
            key = FolderPath

            userDict  = {key:images_list}
            embeddingList = []
            
            model = load_model(modelPath)

            with torch.no_grad():
                for _, images in userDict.items():
                    
                    for image in images:
                            
                        userImage = PIL.Image.open(image)
                        transform1 = transforms.Resize(size=(128,128))
                        transform2 = transforms.ToTensor()
                        transform3 = transforms.Lambda(lambda x: x[:3])

                        userImage = transform1(userImage)
                        userImage = transform2(userImage)
                        userImage = transform3(userImage)
                        
                        
                        userImage = userImage.unsqueeze(dim=0)
                        userImage = userImage.to(DEVICE)
                        embedding = model(userImage)
                        

                        embeddingList.append(embedding)

        
        

        else:
            embeddingList = []

            model = load_model(modelPath)
            
            with torch.no_grad():
                
                
                    userImage =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    userImage = PIL.Image.fromarray(userImage)
                    transform1 = transforms.Resize(size=(128,128)) 
                    transform2 = transforms.ToTensor()
                    transform3 = transforms.Lambda(lambda x: x[:3])

                    userImage = transform1(userImage)
                    userImage = transform2(userImage)
                    userImage = transform3(userImage)
                    
                    
                    userImage = userImage.unsqueeze(dim=0)
                    userImage = userImage.to(DEVICE)
                    embedding = model(userImage)
                    

                    embeddingList.append(embedding)
                
        meanEmbedding = reduce(torch.Tensor.add_,embeddingList,torch.zeros_like(embeddingList[0]))
        torch.div(meanEmbedding,len(embeddingList))
        meanEmbedding = F.normalize(meanEmbedding, p=2, dim=1)
        
        

        if userCreation:
            try:
                path_list = os.listdir("app_data/user_images/")
                
                for user_folder in path_list:
                    user_folder = "app_data/user_images/" + user_folder
                    shutil.rmtree(user_folder)
                
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        return meanEmbedding


def user_creation(userID,frame, count):
            
          
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        
            path = "app_data/user_images" + "/" + userID
        
          
                 
            if not os.path.exists(path):
                os.makedirs(path)
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors = 7, minSize = (100,100), flags = cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in face:
                face = frame[y:y + h, x:x + w]
                filename = path + "/frame%d.jpg" % count
                cv2.imwrite( filename , face)  
                
                time.sleep(0.5)
            
             
            

    
            if userID != None:
                pass
            
           
            
def verify_no_username(embedding, collection, threshold_value):
    # Specify thresholds


    # Load embeddings from file
        results = {}
        for users in collection.find():
            user_name = users['user_name']
            user = users['user_embedding']
            user_embedding = pickle.loads(user)

            result = F.pairwise_distance(embedding, user_embedding,p=2)
            results[user_name] = result
            
        
    
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
        result = min(results.values())
        
        if (result < threshold_value) & (result != 0.0):
            verif_username = min(results, key = results.get)
            verified = True
        else: 
            
            verif_username = "Not Found"
            verified = False
        
        return verified, verif_username


def verify_with_username(embedding, user_name, collection, threshold_value):
    # Specify thresholds


    # Load embeddings from file
        user = get_user_embedding(collection, user_name)
        if user != None:
            user_embedding = pickle.loads(user)

            result = F.pairwise_distance(embedding, user_embedding,p=2)

            if (result < threshold_value) & (result != 0.0):
                verified = True
                verif_username = user_name

            else : 
                verif_username = "No Match"
                verified = False
           
        
        else: 
                verif_username = "UserName Not Found"
                verified = False

        return verified, verif_username

        

def get_collection(db, collection_name):
    '''This method is to get a collection from the database.'''
    a = db[collection_name]
    return a


def get_user_embedding(collection_name, user_name):
    a = collection_name.find_one({"user_name": user_name})
    if a != None:

        return a["user_embedding"]
    else:
        return None

def get_all_usernames(collection_name):
    '''This method is to get all the documents from the database.'''
    userlist = []
    for users in collection_name.find():
        userlist.append(users['user_name'])

    return userlist
        

def delete_user(collection_name, user_name):
    '''This method is to delete a document from the database.'''
    collection_name.delete_one({"user_name": user_name})



def insert_user(collection_name, user_name, user_embedding):
    '''This method is to insert a document into the database.'''
    collection_name.insert_one({"user_name": user_name, "user_embedding": user_embedding})


myclient = MongoClient("mongodb://mongodb:27017/")

mydb = myclient["face_id_app"]

dblist = myclient.list_database_names()
if "face_id_app" in dblist:
  print("The database exists.")

mycol = mydb["users"]

collist = mydb.list_collection_names()
if "users" in collist:
  print("The collection exists.")

tabFace, tabDatabase = st.tabs(["Face Recognition", "Database"])

with tabFace:

    st.title('FACE RECOGNITION')

    user_name = st.text_input(':sunglasses:', "Enter your name")

    img_file_buffer = st.camera_input("Take a picture to access app", key="camera")

    start_face_recognition_no_username = st.checkbox("Access the app without username")

    start_face_recognition_with_username = st.checkbox("Access the app with username")

    create_user_button = st.checkbox("Create user")

    threshold_value = st.slider('Select threshold value (1 is less strict, 0 is more strict): ', 0.0, 1.0, 0.55)




    if img_file_buffer is not None and (start_face_recognition_with_username or start_face_recognition_no_username) and not create_user_button :
    
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)



        st.write("Looking for your face...")
        
        embedding = preprocess(cv2_img)
        

        if type(embedding) != type(None) and start_face_recognition_with_username and not start_face_recognition_no_username:
            st.write("Starting recognition...")
            verified, verif_username = verify_with_username(embedding, user_name, mycol, threshold_value)

            if verified:
                st.success("Welcome %s" % verif_username)
            else:
                st.error("Access not granted")

        elif type(embedding) != type(None) and start_face_recognition_no_username and not start_face_recognition_with_username and img_file_buffer is not None:
            st.write("Starting recognition...")
            verified, verif_username = verify_no_username(embedding, mycol, threshold_value)

            if verified:
                st.success("Welcome %s" % verif_username)
            else:
                st.error("Access not granted")
        else:
            st.error("Face not found")
            
            

    elif create_user_button and not start_face_recognition_no_username:
    
        count = 0
        nb_of_pictures = 10
        while count < nb_of_pictures:
            if img_file_buffer is not None:
                st.write("Taking pictures, turn your head slowly...")
                
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            
                user_creation(user_name, cv2_img, count)
                
                
                count += 1
        
        user_mean_embedding = create_verif_embedding(True,user_name,modelPath=MODELPATH)
        user_embedding = (pickle.dumps(user_mean_embedding))
        insert_user(mycol, user_name, user_embedding)
        
    
        st.success("User " + user_name+ " created")

with tabDatabase:
    st.title('DATABASE')

    see_users = st.button("See all users")

    find_user = st.text_input(':sunglasses:', "Enter user name")

    find_user_button = st.button("Find user")


    if see_users:
        all_users = get_all_usernames(mycol)
        for user in all_users:
            st.write("Username : " , user)


    if find_user:
        all_users = get_all_usernames(mycol)
        if find_user in all_users:
            st.success("User : %s found "%find_user)
        else:
            st.error("User : %s not found "%find_user)


