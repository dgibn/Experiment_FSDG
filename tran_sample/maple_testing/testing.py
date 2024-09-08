import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from model_trainers.lasp import *
import os
import glob 

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

device = "cuda:7" if torch.cuda.is_available() else "cpu"
model1, preprocess = clip.load("ViT-B/32", device='cpu')
model2, preprocess = clip.load("ViT-B/32", device=device)


path_classes='/raid/biplab/divyam/Divyam/office_home_dg/art/train'
domain_name1 = path_classes.split('/')[-1]
class_names=os.listdir(path_classes)
class_names = [i for i in class_names if ".DS" not in i]
class_names.sort()
domain_names=[]
domain_names.append(domain_name1)
domain_names.append(domain_name1)
domain_names.append(domain_name1)

train_classnames = class_names[:6]+ ['unknown']
model = CustomCLIP(train_classnames, model1, model2).to(device)
target_domains = ["Art","Clipart","Real_World","Product"]

# model.load_state_dict(torch.load('/raid/biplab/divyam/Divyam/tran/real_world.pth'))

'''
############### The target dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
# label_class_loc1=[]
label_dom1=[]
path_dom1="/raid/biplab/divyam/Divyam/OfficeHomeDataset_10072016/"+target_domains[3]
dirs_dom1=os.listdir(path_dom1)
class_names = dirs_dom1
num_classes = len(class_names)
dirs_dom1.sort()
c=0
loc=0
index=0
for i in dirs_dom1:
  if index in [0,1,2,3,4,5,6]:
    impaths=path_dom1+'/' +i
    paths=glob.glob(impaths+'*/**.png')
    image_path_dom1.extend(paths)
    label_class_dom1.extend([c for _ in range(len(paths))])
    # label_class_loc1.extend([loc for _ in range(len(paths))])
    # loc=loc+1
  c=c+1
  index=index+1  
label_dom1=[0 for _ in range(len(image_path_dom1))] 

'''
############### Making the test dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)

label_class_final=[]
label_class_dom1_modified = [label if label <= 5 else 6 for label in label_class_dom1]
label_class_final.extend(label_class_dom1_modified)
#label_class_final.extend(label_class_dom1)

label_dom_final=[]
label_dom_final.extend(label_dom1)


class Office_Train(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,num_classes)  
  
    return image, domain, label, label_one_hot 

train_ds=Office_Train(image_path_final,label_dom_final,label_class_final)
print(len(train_ds))
train_dl=DataLoader(train_ds,batch_size=8,num_workers=4, shuffle=True) 


with torch.no_grad():
    
    probs_all = torch.empty(0).to(device)
    labels_all = torch.empty(0).to(device)
    class_all = torch.empty(0).to(device)
    tqdm_object = tqdm(train_dl, total=len(train_dl))
    
    for img, domain, label, label_one_hot  in tqdm_object:
        img = img.to(device)
        domain = domain.to(device)
        label = label.to(device)
        label_one_hot = label_one_hot.to(device)
        
        test_output,_,_= model(img)
        test_prob = F.softmax(test_output, dim=1)
        # print(test_prob)
        # exit()
        max_values, classes = torch.max(test_prob, dim=1)
        # print("max_values", max_values)
        # print("classes", classes)

        probs_all = torch.cat((probs_all, max_values), dim=0)
        labels_all = torch.cat((labels_all, label), dim=0)
        class_all = torch.cat((class_all, classes), dim=0)
# print("probs_all", probs_all.shape)    
# print("labels_all", labels_all.shape)   
# print("class_all", class_all.shape)  
# exit()   
      
confidence = 0.5 
probs_all.shape, labels_all.shape, class_all.shape

class_confidence =  class_all[probs_all<confidence]
label_confidence =  labels_all[probs_all<confidence]
preds_openset_samples = len(class_confidence[label_confidence > 5])
total = sum(labels_all > 5)
print(f'Percentage of correct open set predictions {preds_openset_samples*100/total}')

closed_set = sum(torch.eq(class_all[labels_all <= 5], labels_all[labels_all <= 5]))
total_closed = len(labels_all[labels_all <= 5])
print(f'Percentage of closed set predictions {closed_set*100/total_closed}')