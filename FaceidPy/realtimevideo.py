
# Import kivy UX components


import glob
import torchvision.transforms as transforms
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn

import shutil

import PIL
# Import other dependencies
import cv2
import torch
import os
import numpy as np
import time
DEVICE = 'cuda'
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


        



    

# Load image from file and conver to 100x100px
def preprocess(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors = 7, minSize = (100,100), flags = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in face:
        face = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        embedding = create_verif_embedding(face)

        return embedding

    
   


def create_verif_embedding(frame,modelPath=MODELPATH):
    
    embeddingList = []

    
    model = SiameseNet().to(DEVICE)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    
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


        
        
        
        

    return meanEmbedding



# Verification function to verify person
def verify(embedding, frame):
    # Specify thresholds


    # Load embeddings from file
        results = []
        for user in glob.iglob("/home/hippolyte/Desktop/AICG/FACEID/user_embeddings" + "/*"):
            user_embedding = torch.load(user)
            #userID = user.split("/")[-1].split(".")[0].split("embedding")[-1]
            result = F.pairwise_distance(embedding, user_embedding,p=2)
            results.append(result[0].item())
            
        
    
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
        if min(results) < 0.55:
            detection = results.index(min(results))
            verified = True
        else: 
            detection = None
            verif_username = "Not Found"
            verified = False
        
        usernames = []
        if verified:
            for user in glob.iglob("/home/hippolyte/Desktop/AICG/FACEID/user_embeddings" + "/*"):
                username = user.split("/")[-1].split(".")[0].split("embedding")[-1]
                usernames.append(username)
            verif_username = usernames[detection]
    # Verification Threshold: Proportion of positive predictions / total positive samples 

    # Set verification text 
        if verified:
            cv2.putText(frame, 'UserID : %s' % verif_username , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else: 
            cv2.putText(frame, "Not verified", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

          
       

 


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)


    while True:
        ret, frame = capture.read()
        embedding = preprocess(frame)
        if type(embedding) != type(None):
            verify(embedding, frame)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   



  




    