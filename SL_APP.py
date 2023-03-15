import streamlit as st 
import PIL.Image


import glob
import torchvision.transforms as transforms
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn

import cv2
import torch
import os
import shutil
import time
import numpy as np

if torch.cuda.is_available():
    DEVICE = 'cuda'
# else :
#      DEVICE = 'cpu'
MODELPATH = "trained_model.pth"
MODEL2 = "trained_model2.pth"

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
            
           
            
def verify(embedding):
    # Specify thresholds


    # Load embeddings from file
        results = []
        for user in glob.iglob("user_embeddings" + "/*"):
            user_embedding = torch.load(user)
            #userID = user.split("/")[-1].split(".")[0].split("embedding")[-1]
            result = F.pairwise_distance(embedding, user_embedding,p=2)
            results.append(result[0].item())
            
        
    
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
        result = min(results, default=0)
        if (result < 0.55) & (result != 0.0):
            detection = results.index(min(results))
            verified = True
        else: 
            detection = None
            verif_username = "Not Found"
            verified = False
        
        usernames = []
        if verified:
            for user in glob.iglob("user_embeddings" + "/*"):
                username = user.split("/")[-1].split(".")[0].split("embedding")[-1]
                usernames.append(username)
            verif_username = usernames[detection]
    # Verification Threshold: Proportion of positive predictions / total positive samples 

        return verified, verif_username



st.title('FACE RECOGNITION APP')

user_name = st.text_input(':sunglasses:', "Enter your name")

img_file_buffer = st.camera_input("Take a picture to acess app", key="camera")

start_face_recognition = st.checkbox("Access the app")

create_user_button = st.checkbox("Create user")


if img_file_buffer is not None and start_face_recognition and not create_user_button:
 
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)



    st.write("Looking for your face...")
    
    embedding = preprocess(cv2_img)
    

    if type(embedding) != type(None):
        st.write("Starting recognition...")
        verified, verif_username = verify(embedding)

        if verified:
            st.success("Welcome %s" % verif_username)
        else:
            st.error("Access not granted")
    else:
         st.error("Face not found")
        
        

elif create_user_button and not start_face_recognition:
   
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
    torch.save(user_mean_embedding, "user_embeddings/" + "embedding" + user_name + ".pt")
    st.success("User " + user_name+ " created")
