import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip_old import clip
from model_trainers.stylip import *
import os
import glob 
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

device = "cuda:2" if torch.cuda.is_available() else "cpu" 
model1, preprocess = clip.load("ViT-B/32", device='cpu')
model2, preprocess = clip.load("ViT-B/32", device=device)

domains = ['clipart', 'painting', 'sketch', 'real']

cla = 3
shots = 5
print(shots)
print(domains[cla])

path_classes='/raid/biplab/divyam/Divyam/domainnet/'+domains[cla]
domain_name1 = path_classes.split('/')[-1]
class_names=os.listdir(path_classes)  
class_names.sort()
# print(class_names)
splits_1 = list(range(0,20)) + list(range(40,60))
splits_2 = list(range(0,10)) + list(range(20,40)) + list(range(80,90))
splits_3 = list(range(10,20)) + list(range(40,50)) + list(range(60,80))
train_classes =list(set().union(splits_1, splits_2, splits_3))
train_classnames =[class_names[i] for i in train_classes]
model = CustomCLIP(train_classnames, model1).to(device)

model.load_state_dict(torch.load('/raid/biplab/divyam/Divyam/tran/output/nssl/minidomainnet/stylip_{}_{}.pth'.format(domains[cla],shots)))

'''
############### The target dataset 1 ##################
'''

test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_domain_names=[]
test_path_dom='/raid/biplab/divyam/Divyam/domainnet/'+domains[cla]
test_domain_name = test_path_dom.split('/')[-1]
testdom_classnames = []
test_filenames = []
with open('/raid/biplab/divyam/Divyam/splits_mini/{}_test.txt'.format(domains[cla]), 'r') as file:
    for line in file:
        parts = line.strip().split('/')
        if len(parts) >= 3:
            filename = parts[-1].split()[0]
            class_name = parts[-2]
            if class_name not in testdom_classnames:
                testdom_classnames.append(class_name)            
            filename_with_class_name = f"{class_name}/{filename}"
            test_filenames.append(filename_with_class_name)
testdom_classnames.sort()
test_classnames = []
print(len(test_filenames))
c=0
index=0
test_index = list(range(0,5))+list(range(8,18))+list(range(25,35))+list(range(43,48))+list(range(75,80))+list(range(83,88))+list(range(90,126))

for i in testdom_classnames:
    # if index in index_dom1:
    if index in test_index:
        paths = [filename for filename in test_filenames if filename.startswith(i + '/')]
        random.shuffle(paths)
        for j in paths:
            selected_paths = glob.glob(test_path_dom + '/' + j)
            test_image_path_dom.extend(selected_paths)
            test_label_class_dom.extend([c for _ in range(len(selected_paths))])
    c = c + 1
    index = index+1
test_label_dom=[3 for _ in range(len(test_image_path_dom))]
  
test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)

test_label_class_final=[]
test_label_class_final.extend(test_label_class_dom)
# test_label_class_final_modified = [label if label <= 89 else 90 for label in test_label_class_dom]
# test_label_class_final.extend(test_label_class_final_modified)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)


test_domain_names.append(test_domain_name)

'''
############### Making the test dataset ##################
'''   
  
# image_path_final=[]
# image_path_final.extend(image_path_dom1)
# label_class_final=[]
# # label_class_dom1_modified = [label if label <= 64 else 65 for label in label_class_dom1]
# # label_class_final.extend(label_class_dom1_modified)
# label_class_final.extend(label_class_dom1)
# label_dom_final=[]
# label_dom_final.extend(label_dom1)


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
 
    label_one_hot=F.one_hot(label,len(testdom_classnames))  
  
    return image, domain, label, label_one_hot 

train_ds=Office_Train(test_image_path_final,test_label_dom_final,test_label_class_final)
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
        test_output = model(img)
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
preds_openset_samples = len(class_confidence[label_confidence > 89])
total = sum(labels_all > 89)
print("total:",total)
openset_pred = preds_openset_samples*100/total
print(f'Percentage of correct open set predictions {openset_pred}')

closed_set = sum(torch.eq(class_all[labels_all <= 89], labels_all[labels_all <= 89]))
total_closed = len(labels_all[labels_all <= 89])
print("total_closed:",total_closed)
closedset_pred = closed_set*100/total_closed
print(f'Percentage of closed set predictions {closedset_pred}')

hms = (2*closedset_pred*openset_pred)/(closedset_pred+openset_pred)
print("Harmonic Score:",hms)  