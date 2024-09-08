import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from ptflops import get_model_complexity_info

from clip import clip
from clip_old import clip as CLIP
from model_trainers.maple import *
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
from maple_testing import testing


def load_clip_to_cpu():
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model
  

device = "cuda:5" if torch.cuda.is_available() else "cpu"
clip_model = load_clip_to_cpu()
_, preprocess = CLIP.load("ViT-B/32", device=device)

############################################################
#################-------DATASET------#######################
############################################################

target_domains = ["Art","Clipart","Real_World","Product"]

'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
path_dom1="/raid/biplab/divyam/Divyam/OfficeHomeDataset_10072016/"+target_domains[0]
dirs_dom1=os.listdir(path_dom1)
# class_names = dirs_dom1
# num_classes = len(class_names)
# class_names.sort()
dirs_dom1.sort()
c=0
# source_images_per_class=100
for i in dirs_dom1:
  impaths = path_dom1 + '/' + i
  class_names1.append(i)
  paths = glob.glob(impaths + '/**.jpg')
  random.shuffle(paths)
  #selected_paths = paths[:source_images_per_class]
  image_path_dom1.extend(paths)
  label_class_dom1.extend([c for _ in range(len(paths))])
  c = c + 1
label_dom1=[0 for _ in range(len(image_path_dom1))] 


'''
############### The source dataset 2 ##################
'''

image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
path_dom2="/raid/biplab/divyam/Divyam/OfficeHomeDataset_10072016/"+target_domains[1]
dirs_dom2=os.listdir(path_dom2)
dirs_dom2.sort()
c=0
for i in dirs_dom2:
  class_names2.append(i)
  impaths=path_dom2+'/' +i
  paths=glob.glob(impaths+'*/**.jpg')
  random.shuffle(paths)
  #selected_paths = paths[:source_images_per_class]
  image_path_dom2.extend(paths)
  label_class_dom2.extend([c for _ in range(len(paths))])
  c=c+1
label_dom2=[1 for _ in range(len(image_path_dom2))]  


'''
############### The source dataset 3 ##################
'''

image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
path_dom3="/raid/biplab/divyam/Divyam/OfficeHomeDataset_10072016/"+target_domains[2]
dirs_dom3=os.listdir(path_dom3)
dirs_dom3.sort()
c=0
for i in dirs_dom3:
  class_names3.append(i)
  impaths=path_dom3+'/' +i
  paths=glob.glob(impaths+'*/**.jpg')
  random.shuffle(paths)
  #selected_paths = paths[:source_images_per_class]
  image_path_dom3.extend(paths)
  label_class_dom3.extend([c for _ in range(len(paths))])
  c=c+1
label_dom3=[2 for _ in range(len(image_path_dom3))]

'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)

'''
############### Test dataset ##################
'''

test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_path_dom="/raid/biplab/divyam/Divyam/OfficeHomeDataset_10072016/"+target_domains[3]
test_dirs_dom=os.listdir(test_path_dom)
test_class_names = test_dirs_dom
test_num_classes = len(test_class_names)
test_class_names.sort()
test_dirs_dom.sort()


class Office_Train(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)
  
  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx])).to(device)
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,test_num_classes)  
  
    return image, domain, label, label_one_hot 

train_ds=Office_Train(image_path_final,label_dom_final,label_class_final)
print("length of train_ds", len(train_ds))
train_dl=DataLoader(train_ds,batch_size=1, shuffle=True) 
img, domain, label, label_one_hot = next(iter(train_dl))


'''
############### Making and trainning the model ##################
'''   


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
      
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img, domain, label, label_one_hot  in tqdm_object:
        img = img.to(device)
        domain = domain.to(device)
        label = label.to(device)
        label_one_hot = label_one_hot.to(device)
        output = model(img)
        loss = F.cross_entropy(output, label)
        if loss.isnan():
          continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)
        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, accuracy= compute_accuracy(output, label)[0].item(), lr=get_lr(optimizer))
    return loss_meter
  
test_class_names.sort()
train_classnames = test_class_names[:48]

model = CustomCLIP(train_classnames, clip_model = clip_model).to(device)
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

params = [
            {"params": model.prompt_learner.parameters()},
        ]

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_no = sum([np.prod(p.size()) for p in model_parameters])
print(params_no)

with torch.cuda.device(0):
  macs, params_num = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params_num))
exit()

optimizer = torch.optim.SGD(params, lr=0.0026)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
scaler = GradScaler() 


step = "epoch"
best_loss = float('inf')
prev_new_acc = 0
for epoch in range(10):
  print(f"Epoch: {epoch + 1}")
  model.train()
  train_loss = train_epoch(model, train_dl, optimizer, lr_scheduler, step)


testing(model, "sketch")
  
MODEL_PATH = Path("/raid/biplab/divyam/Divyam/tran")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "product.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
        f=MODEL_SAVE_PATH)