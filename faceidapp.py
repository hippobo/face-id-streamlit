from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import PIL.Image
# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput

import glob
import torchvision.transforms as transforms
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn
# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
import shutil

# Import other dependencies
import cv2
import torch
import tensorflow as tf
import os
import numpy as np
import time
DEVICE = 'cuda'
MODELPATH = "trained_model2.pth"
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


class CamApp(App):

    def build(self):
        
        # Main layout components 
        self.web_cam = Image(size_hint=(1,1))
        self.verifButton = Button(text="Verify", on_press=self.verify,
         size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1), color = (1,0,0,1))
        self.userID_label = Label(text = "User ID : ", size_hint=(1,.1))
        Window.size = (800, 800)
        

        

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.userID_label)
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.verifButton)
        self.username = TextInput(text="Enter Username", size_hint=(1,.1), multiline=False)
        layout.add_widget(self.username)
        self.create_user_comp = Button(text="Create User", on_press=self.user_creation,
         size_hint=(1,.1))
        layout.add_widget(self.create_user_comp)

        





        
        self.model = torch.load(MODELPATH, map_location=torch.device('cpu'))


        # # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/60.0)
        
        return layout


    # # Run continuously to get webcam feed


    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        #frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, userCreation,userID = None):
        vidcap = self.capture
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0
        if userCreation:
            path = "/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/user_images" + "/" + userID
            nbOfPictures = 10
        else:
            path = "/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/verif_images"
            nbOfPictures = 1 
            
            
        
        if not os.path.exists(path):
            os.makedirs(path)

        
        while (vidcap.isOpened()):
            
            ret, frame = vidcap.read()
            frameRate = vidcap.get(5)
            if(ret and count < nbOfPictures):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors = 7, minSize = (100,100), flags = cv2.CASCADE_SCALE_IMAGE)
                for (x, y, w, h) in face:
                    face = frame[y:y + h, x:x + w]
                    filename = path + "/frame%d.jpg" % count
                    cv2.imwrite( filename , face)    
                count = count + 1
                time.sleep((int(frameRate)//int(frameRate)) * 1.5)
            else: 
                break


    
        if userID != None:
            self.verification_label.text = "User %s Created" % userID
            self.verification_label.color = (0,1,0,1)
        
            


    def create_verif_embedding(self, userCreation,userID = None,modelPath=MODELPATH):
        
        images_list = []
        if userCreation:
            FolderPath = "/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/user_images/" + userID
        else: 
            FolderPath = "/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/verif_images"
            if len(os.listdir("/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/verif_images")) == 0:
                
                return None
        for picture in glob.iglob(f'{FolderPath}/*'):
                images_list.append(picture)

        

        
        key = FolderPath

        userDict  = {key:images_list}
        embeddingList = []
        
        model = SiameseNet().to(DEVICE)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        
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
                
            
            meanEmbedding = reduce(torch.Tensor.add_,embeddingList,torch.zeros_like(embeddingList[0]))
            torch.div(meanEmbedding,len(embeddingList))
            meanEmbedding = F.normalize(meanEmbedding, p=2, dim=1)
        


        if not userCreation:
            Verif_Images = glob.glob("/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/verif_images" + "/*")

            for f in Verif_Images:
                os.remove(f)
        else:
            try:
                path_list = os.listdir("/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/user_images/")
                
                for user_folder in path_list:
                    user_folder = "/home/hippolyte/Desktop/AICG/FACEIDAPI/app_data/user_images/" + user_folder
                    shutil.rmtree(user_folder)
                
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
           
            
            
            
            

        return meanEmbedding


    def user_creation(self, userID, modelPath=MODELPATH):
            userID = self.username.text
            pathOut = "/home/hippolyte/Desktop/AICG/FACEIDAPI/user_embeddings/"
        
            self.preprocess(userCreation=True,userID=userID)
            user_mean_embedding = self.create_verif_embedding(True,userID,modelPath)
            torch.save(user_mean_embedding, pathOut + "embedding" + userID + ".pt")


    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        
        self.preprocess(userCreation=False)
        verif_embedding = self.create_verif_embedding(userCreation = False)
        if verif_embedding == None:
                self.verification_label.text = 'Cannot find your face in the frame'
                self.verification_label.color = (1,0,0,1)

        else: # Build results array
            results = []
            for user in glob.iglob("/home/hippolyte/Desktop/AICG/FACEIDAPI/user_embeddings" + "/*"):
                user_embedding = torch.load(user)
                #userID = user.split("/")[-1].split(".")[0].split("embedding")[-1]
                result = F.pairwise_distance(verif_embedding, user_embedding,p=2)
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
                for user in glob.iglob("/home/hippolyte/Desktop/AICG/FACEIDAPI/user_embeddings" + "/*"):
                    username = user.split("/")[-1].split(".")[0].split("embedding")[-1]
                    usernames.append(username)
                verif_username = usernames[detection]
        # Verification Threshold: Proportion of positive predictions / total positive samples 

        # Set verification text 
            if verified:
                self.verification_label.text = 'Verified: Access granted'
                self.verification_label.color = (0,1,0,1)
            else: 
                self.verification_label.text = 'Unverified, access denied'
                self.verification_label.color = (1,0,0,1)
            self.userID_label.text = 'User ID: ' + str(verif_username)

        # Log out details
            Logger.info(min(results))
            Logger.info(detection)
            Logger.info(verified)

        
            return results, verified



if __name__ == '__main__':
    CamApp().run()