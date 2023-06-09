# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import reduce

import glob
import os
import random
import time
from cProfile import label
from re import M
from torch.optim.lr_scheduler import StepLR
import cv2
import os

import torch

import torch.nn.functional as F



import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image
from torch import Tensor
from torch.autograd import Variable
from torch.nn import (Conv2d, Dropout, Flatten, Linear, MaxPool2d, Module,
                      ReLU, Sequential)
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.loss import MarginRankingLoss, TripletMarginLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset



BATCH_SIZE = 256
LR = 0.001
EPOCHS = 750
MARGIN = 1.0

DEVICE = 'cuda'

def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, text = None, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    plt.text(0.5, 0.5, text , size=20,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )

    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()



def imshow(img, text=None):
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.show()

class SiameseDataset(Dataset):

    def __init__(self,trainBool, transform = None):
        
        self.trainBool = trainBool
        self.transform = transform
        
        if self.trainBool:

            self.root_dir = "AllFacesTrain"
            

        else:
            self.root_dir = "FacesTest"

        classes_list = []
       
        for classes in glob.iglob(f'{self.root_dir}/*'):
            images_list = []
            for image in glob.iglob(classes+'/*[png][jpg]'):
                    images_list.append(image)
            classes_list.append(images_list)

    
        keys = list(range(len(classes_list)))

        self.dataDict  = dict(zip(keys,classes_list))
        
        
    def __getitem__(self, index):
  

        if self.trainBool:
            list_keys = self.dataDict.keys()

            indClass = np.random.randint(0,len(list_keys))
            
            classSelect = list(range(len(list_keys)))
            indClass = np.random.randint(0,len(list_keys))
            classSelect.remove(indClass)

            a,p = np.random.choice(self.dataDict[indClass],size=2,replace=False)
            
            
            anchor_label = indClass + 1
           
            indClassN = random.choice(classSelect)
            n = np.random.choice(self.dataDict[indClassN],size=1,replace=False)
            

        else:
            list_keys = self.dataDict.keys()
           
            indClass = np.random.randint(0,len(list_keys))
            
            classSelect = list(range(len(list_keys)))
            indClass = np.random.randint(0,len(list_keys))
            classSelect.remove(indClass)
            
            a,p = np.random.choice(self.dataDict[indClass],size=2,replace=False)
            
            anchor_label = indClass + 1
            
            indClassN = random.choice(classSelect)
            n = np.random.choice(self.dataDict[indClassN],size=1,replace=False)
            


        a = Image.open(a)
        p = Image.open(p)
        n = Image.open(n[0])


        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        
    
        return a,p,n, anchor_label

    
    def __len__(self):
        list_values = self.dataDict.values()
       
        return len(list_values)

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

class TripletLossFunction(nn.Module):


    def __init__(self,margin):
        super(TripletLossFunction, self).__init__()
        self.margin = margin
        self.dist = PairwiseDistance(p=2)

    def forward(self, a, p , n):
        positive_distance = self.dist.forward(a,p)
        negative_distance = self.dist.forward(a,n)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()

class L1Dist(nn.Module):


    def __init__(self):
        super(L1Dist, self).__init__()
        self.dist = PairwiseDistance(p=2)

    def forward(self, userMeanEmbedding, validEmbedding):
        distance = self.dist.forward(userMeanEmbedding,validEmbedding)
        return distance[0].item()
     

    

def trainModel():

        
    triplet_train_dataset = SiameseDataset(trainBool=True, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Lambda(lambda x: x[:3])]))
    triplet_test_dataset = SiameseDataset(trainBool=False, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Lambda(lambda x: x[:3])]))

    train_dataloader = DataLoader(triplet_train_dataset, shuffle=False, num_workers=2, batch_size = BATCH_SIZE)
    test_dataloader = DataLoader(triplet_test_dataset, shuffle=True, num_workers=2, batch_size = BATCH_SIZE)






    batch = next(iter(train_dataloader))

    data, labels = batch
    concatenated = torch.cat((data[0], data[1], data[2]),0)
    imshow(torchvision.utils.make_grid(concatenated, nrow=BATCH_SIZE), text=(labels[0],labels[1],labels[2]))

    model = SiameseNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    criterion = TripletLossFunction(margin = MARGIN) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50, gamma = 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)




    fig = plt.figure(figsize=(20, 15), facecolor="azure")
    ax  = fig.add_subplot( 111 )

    model.train()
    min_loss = 10.0
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        running_loss = []
        

        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_dataloader):
            anchor_img, positive_img, negative_img,anchor_label =anchor_img.to(DEVICE), positive_img.to(DEVICE), negative_img.to(DEVICE), anchor_label.to(DEVICE)
            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            
            loss = criterion(anchor_out, positive_out, negative_out)
        
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss.append(loss.cpu().detach().numpy())
            
            if np.mean(running_loss) < min_loss:
              min_loss = np.mean(running_loss)
              torch.save(model.state_dict(),"trained_model4.pth")

        print("best loss: ", min_loss)

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, EPOCHS, np.mean(running_loss)))
        print("Seconds since epoch =", time.time()- start_time)	

        if epoch % 4 == 0 and epoch > 1:
            print("*******TESTING********")

            test_results = []
            labels = []

            model.eval()
            with torch.no_grad():
                for (a, p, n, label) in (test_dataloader):
                    
                    test_results.append(model(a.to()).cpu().numpy())
                    labels.append(label)
                    
            test_results = np.concatenate(test_results)
            labels = np.concatenate(labels)
            test_results.shape
            
            for label in np.unique(labels): 
                tmp = test_results[labels==label]
                ax.scatter(tmp[:, 0], tmp[:, 1], label=label)


            fig.legend()
            fig.savefig('test.png')
            ax.clear()
            model.train()
            
            
    torch.save(model.state_dict(),
            "trained_model.pth")



    train_results = []

    labels = []
    model = SiameseNet()
    model.load_state_dict(torch.load("trained_model.pth"))
    model.eval()
    with torch.no_grad():
        for img, _, _, label in (train_dataloader):
            train_results.append(model(img.to()).cpu().numpy())
            labels.append(label)
            
    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    train_results.shape
    plt.figure(figsize=(20, 15), facecolor="azure")
    for label in np.unique(labels): 
        tmp = train_results[labels==label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label)


    plt.legend()
    plt.savefig('train.png')
    plt.show()

def img_to_encoding(imagePath,modelPath= "trained_model.pth") -> torch.Tensor:
    model = SiameseNet().to(DEVICE)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    with torch.no_grad():
        image = imagePath
        image = Image.open(image)
        transform1 = transforms.Resize(size=(128,128))
        transform2 = transforms.ToTensor()
        transform3 = transforms.Lambda(lambda x: x[:3])
        image = transform1(image)
        image = transform2(image)
        image = transform3(image)

        image = image.unsqueeze(dim=0)
        image = image.to(DEVICE)
        embedding = model(image)
        
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

def create_user(userId,modelPath="trained_model.pth"):
    images_list = []
    usersFolderPath = "/home/hippolyte/Desktop/AICG/FACEID/UserImages/%i" %userId
    for picture in glob.iglob(f'{usersFolderPath}/*'):
            images_list.append(picture)
            

    
    key = usersFolderPath

    userDict  = {key:images_list}
    embeddingList = []
    
    model = SiameseNet().to(DEVICE)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
     
    with torch.no_grad():
        for label, images in userDict.items():
            
            userLabel = label
            for image in images:
                    
                userImage = Image.open(image)
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
       



        
        
        
        

    return userLabel, meanEmbedding


def extractuserImages(pathOut, userID,nbOfPictures):
    vidcap = cv2.VideoCapture(0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    path = pathOut + "/%i" % userID
    
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
            vidcap.release()


    
    
    
   
    
    print("User Images %i Done" % userID)



if __name__=="__main__":

    trainModel()








