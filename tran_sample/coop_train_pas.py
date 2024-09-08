import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from collections import OrderedDict
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from model_trainers.stylip import *
# from model_trainers.coop_slip import *
import os
import glob 
import utils
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import random
import math

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

device = "cuda:2" if torch.cuda.is_available() else "cpu"
Clipmodel, preprocess = clip.load("ViT-B/32", device="cpu")

# Slipmodel = SLIP_VITB16()

############################################################
#################-------DATASET------#######################
############################################################

target_domains =  ['art_painting', 'cartoon', 'photo', 'sketch']
shots=1
'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
path_dom1="/raid/biplab/divyam/Divyam/pacs_data/pacs_data/" +target_domains[1]
dirs_dom1=os.listdir(path_dom1)
splits_1 = [3, 0, 1]
dirs_dom1.sort()
c=0
source_images_per_class=shots
for i in splits_1:
  impaths = path_dom1 + '/' + dirs_dom1[i]
  class_names1.append(dirs_dom1[i])
  paths = glob.glob(impaths + '/**.jpg')+glob.glob(impaths+'*/**.png')
  random.shuffle(paths)
  paths = paths[:source_images_per_class]
  image_path_dom1.extend(paths)
  label_class_dom1.extend([i for _ in range(len(paths))])
  c+=1
label_dom1=[0 for _ in range(len(image_path_dom1))] 
print(len(label_dom1))

'''
############### The source dataset 2 ##################
'''

image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
path_dom2="/raid/biplab/divyam/Divyam/pacs_data/pacs_data/" +target_domains[2]
dirs_dom2=os.listdir(path_dom2)
dirs_dom2.sort()
splits_2 = [4, 0, 2]
c=0
for i in splits_2:
  class_names2.append(dirs_dom2[i])
  impaths=path_dom2+'/' +dirs_dom2[i]
  paths=glob.glob(impaths+'*/**.jpg')+glob.glob(impaths+'*/**.png')
  random.shuffle(paths)
  paths = paths[:source_images_per_class]
  image_path_dom2.extend(paths)
  label_class_dom2.extend([i for _ in range(len(paths))])
  c=c+1
label_dom2=[1 for _ in range(len(image_path_dom2))]  
print(len(label_dom2))

'''
############### The source dataset 3 ##################
'''

image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
path_dom3="/raid/biplab/divyam/Divyam/pacs_data/pacs_data/" +target_domains[3]
dirs_dom3=os.listdir(path_dom3)
dirs_dom3.sort()
splits_3 = [5, 1, 2]
for i in splits_3:
  class_names3.append(dirs_dom3[i])
  impaths=path_dom3+'/' +dirs_dom3[i]
  paths=glob.glob(impaths+'*/**.jpg')+glob.glob(impaths+'*/**.png')
  random.shuffle(paths)
  paths = paths[:source_images_per_class]
  image_path_dom3.extend(paths)
  label_class_dom3.extend([i for _ in range(len(paths))])
  c=c+1
label_dom3=[2 for _ in range(len(image_path_dom3))]
print(len(label_dom3))


'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
# image_path_final.extend(image_path_dom4)
# image_path_final.extend(image_path_dom5)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
# label_class_final.extend(label_class_dom4)
# label_class_final.extend(label_class_dom5)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
# label_dom_final.extend(label_dom4)
# label_dom_final.extend(label_dom5)

test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_path_dom="/raid/biplab/divyam/Divyam/pacs_data/pacs_data/" +target_domains[0] 
test_dirs_dom=os.listdir(test_path_dom)
test_class_names = test_dirs_dom
num_classes = len(test_class_names)
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
    img =Image.open(self.image_path[idx])
    image = preprocess(img).to(device)
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,num_classes)  
  
    return image,domain, label, label_one_hot

train_ds=Office_Train(image_path_final,label_dom_final,label_class_final)
print(len(train_ds))
train_dl=DataLoader(train_ds,batch_size=4, shuffle=True) 
img,domain, label, label_one_hot = next(iter(train_dl))


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
    
class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, aug1,aug2):
        q_a = aug1
        q_b = aug2

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}

# class SIMCLR(nn.Module):
#     def __init__(self):
#         super().__init__()
#         ssl_mlp_dim=4096
#         ssl_emb_dim=256

#         self.vision_width = 768
#         self.visual = timm.create_model('vit_base_patch16_224', num_classes=0)

#         self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

#     def _build_mlp(self, in_dim, mlp_dim, out_dim):
#         return nn.Sequential(OrderedDict([
#             ("layer1", nn.Linear(in_dim, mlp_dim)),
#             ("bn1", nn.SyncBatchNorm(mlp_dim)),
#             ("relu1", nn.ReLU(inplace=True)),
#             ("layer2", nn.Linear(mlp_dim, mlp_dim)),
#             ("bn2", nn.SyncBatchNorm(mlp_dim)),
#             ("relu2", nn.ReLU(inplace=True)),
#             ("layer3", nn.Linear(mlp_dim, out_dim)),
#         ]))

#     def encode_image(self, image):
#         x = self.visual(image)

#         return x

#     def forward(self, aug1, aug2):
#         h1 = self.visual(aug1)
#         h2 = self.visual(aug2)

#         aug1_embed = self.image_mlp(h1)
#         aug2_embed = self.image_mlp(h2)

#         return {'aug1_embed': aug1_embed,
#                 'aug2_embed': aug2_embed}

SIMCLRloss = SIMCLRLoss().to(device)
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for img,domain, label, label_one_hot  in tqdm_object:
        count = img.size(0)
        img =img.to(device)
        domain = domain.to(device)
        label = label.to(device)
        label_one_hot = label_one_hot.to(device)
        output= model(img)
        # loss_simclr = SIMCLRloss(augment1,augment2)
        loss = F.cross_entropy(output, label) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)
        loss_meter.update(loss.item(), count)
        acc = compute_accuracy(output, label)[0].item() 
        accuracy_meter.update(acc, count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, accuracy= accuracy_meter.avg, lr=get_lr(optimizer))
    return loss_meter, accuracy_meter.avg
  

train_classes =list(set().union(splits_1,splits_2,splits_3))

classnames =list(set().union(class_names1,class_names2,class_names3))
train_classnames = [classnames[i] for i in train_classes]
model = CustomCLIP(train_classnames, Clipmodel).to(device)
for name, param in model.named_parameters():
    if "prompt_learner" not in name:
        param.requires_grad_(False)

params = [
            {"params": model.prompt_learner.parameters()},
        ]
 
optimizer = torch.optim.AdamW(params, weight_decay=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
scaler = GradScaler() 


step = "epoch"
best_loss = float('inf')
prev_new_acc = 0
for epoch in range(5):
  
  print(f"Epoch: {epoch + 1}")
  model.train()
  train_loss, train_acc = train_epoch(model, train_dl, optimizer, lr_scheduler, step)
  
  
MODEL_PATH = Path("/raid/biplab/divyam/Divyam/tran/output/nssl/pacs")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "stylip_art_painting_{}.pth".format(shots)
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
        f=MODEL_SAVE_PATH)
  