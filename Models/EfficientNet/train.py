import os
import re
import random
from datetime import date
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import time

import sklearn
from sklearn import model_selection as sk_model_selection
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import nibabel as nib
import matplotlib.pyplot as plt

from efficientnet_pytorch_3d import EfficientNet3D


import argparse

parser = argparse.ArgumentParser(description='U-Net Classification Train')
parser.add_argument('--mri_type', type=str,
                    help='Train your model on which MRI type. Should be one of: flair,t1,t1ce,t2. All (All means sequentially training the above 4 mri types)', default='flair')
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=2)
parser.add_argument('--base_lr', type=float,
                    help='Base Learning Rate', default=1e-5)
parser.add_argument('--scale_size', type=int,
                    help='Volume Scale Size', default=90)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

RAW_DATA_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_Prep_Segmentation'
MASKS_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_Masks_Train'
LABEL_PATH = './train_labels.csv'
BASE_LR = args.base_lr
SCALE_SIZE = args.scale_size
SAVE_FOLDER = './Model'
NETWORK = 'EfficientNet'

MRI_TYPES = ['flair','t1','t1ce','t2'] if args.mri_type == 'All' else [args.mri_type]

# Set Global Randdom State as 42  
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True



# ============ Helpful Functions ==============    

def load_and_split_labels(label_path):
    """
    Load Training Label and split
    """
    train_df = pd.read_csv(label_path,dtype = {'BraTS21ID':'str','MGMT_value':'int'})
    index_name = train_df[(train_df['BraTS21ID'] == '00109') | (train_df['BraTS21ID'] == '00123') | (train_df['BraTS21ID'] == '00709')].index
    train_df = train_df.drop(index_name).reset_index(drop=True)
        


    df_train, df_valid = sk_model_selection.train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_df["MGMT_value"],
    )
    
    print(f'Number of training samples: {len(df_train)}. Number of valid samples: {len(df_valid )}')
    return df_train,df_valid


def load_raw_voxel(patient_id,mri_type):
    # Normalize voxel volume to 0~255
    voxels = nib.load(f'{RAW_DATA_PATH}/BraTS2021_{patient_id}/BraTS2021_{patient_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    _min = voxels.min()
    _max = voxels.max()
    new_voxels = (voxels - _min) / (_max-_min) * 255.0
    return new_voxels

def load_mask(patient_id):
    return nib.load(f'{MASKS_PATH}/BraTS2021_{patient_id}.nii.gz').get_fdata().astype('float')

def non_0_voxel_mask(voxel,mask):
    length = mask.shape[2]
    start_id = 0
    end_id = length-1

    # From begining to find start index
    for i in range(length):
        if np.max(mask[:,:,i]) != 0:
            start_id = i
            break

    # From final to find end index
    for i in range(length-1,-1,-1):
        if np.max(mask[:,:,i]) != 0:
            end_id = i
            break
    non_0_indexs = slice(start_id,end_id+1)
    
    return voxel[:,:,non_0_indexs],mask[:,:,non_0_indexs]

def find_largest_countours(contours):
    max_cnt = max(contours, key=lambda cnt: cv2.contourArea(cnt))
    return max_cnt


def get_area_over_image_ratio(image, mask):
    _, image_thresh = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    
    # image_contours, _ = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    image_contours, _ = cv2.findContours(image=image_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    if not image_contours:
        return 0
    max_image_cnt = find_largest_countours(image_contours)
    
    _, mask_thresh = cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
    mask_contours, _ = cv2.findContours(image=mask_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    count_n_mask_contours = len(mask_contours)
    if(count_n_mask_contours == 0):
        return 0
    max_mask_cnt = find_largest_countours(mask_contours)
    
    area_mask_over_image_ratio = cv2.contourArea(max_mask_cnt) / cv2.contourArea(max_image_cnt)
    return area_mask_over_image_ratio 

def construct_target_volume(scan_id,mri_type,scale_size=90):
    raw_voxel = load_raw_voxel(scan_id,mri_type)
    mask = load_mask(scan_id)
    
    non_0_voxel,non_0_mask = non_0_voxel_mask(raw_voxel,mask)
    
    mask_WT = non_0_mask.copy()
    mask_WT[(mask_WT == 2) | (mask_WT == 4)] = 1
    voxel_WT = mask_WT*non_0_voxel
    
    length = voxel_WT.shape[2]
    #------------------------------------Warning To-do ------------------
    if scan_id =='00651' and mri_type =='t1ce':
        max_slice_index = length//2
        half_len = scale_size//6

        start_ind = max_slice_index - half_len
        end_ind = max_slice_index + half_len-1
        if end_ind > (length-1):
            diff = end_ind - (length-1)
            start_ind = start_ind-diff
            end_ind = end_ind -diff

        constructed_voxel= np.stack([non_0_voxel[:,:,start_ind],mask_WT[:,:,start_ind],non_0_mask[:,:,start_ind]],axis=-1)

        for i in range(start_ind+1,end_ind+1):
            new_images = np.stack([non_0_voxel[:,:,i],mask_WT[:,:,i],non_0_mask[:,:,i]],axis=-1)
            constructed_voxel = np.dstack((constructed_voxel,new_images))
        
        return constructed_voxel
    #---------------------------------------------------------------------------
    
    if length<(scale_size/3):
        constructed_voxel= np.stack([non_0_voxel[:,:,0],mask_WT[:,:,0],non_0_mask[:,:,0]],axis=-1)
        for i in range(1,length):
            new_images = np.stack([non_0_voxel[:,:,i],mask_WT[:,:,i],non_0_mask[:,:,i]],axis=-1)
            constructed_voxel = np.dstack((constructed_voxel,new_images))
        
        
        fixed_voxel = np.zeros((240,240,scale_size))
        real_length = constructed_voxel.shape[2]
        
        start = (scale_size-real_length)//2
        end = start+real_length
        fixed_voxel[:,:,start:end] = constructed_voxel
        
        return fixed_voxel
    else:
        max_ratio = 0
        max_slice_index = length//2
        for i in range(length):
            current_ratio = get_area_over_image_ratio(non_0_voxel[:,:,i].astype('uint8'), mask_WT[:,:,i].astype('uint8'))
            if current_ratio>max_ratio:
                max_ratio = current_ratio
                max_slice_index = i

        half_len = scale_size//6

        start_ind = max_slice_index - half_len
        end_ind = max_slice_index + half_len-1
        if end_ind > (length-1):
            diff = end_ind - (length-1)
            start_ind = start_ind-diff
            end_ind = end_ind -diff

        constructed_voxel= np.stack([non_0_voxel[:,:,start_ind],mask_WT[:,:,start_ind],non_0_mask[:,:,start_ind]],axis=-1)

        for i in range(start_ind+1,end_ind+1):
            new_images = np.stack([non_0_voxel[:,:,i],mask_WT[:,:,i],non_0_mask[:,:,i]],axis=-1)
            constructed_voxel = np.dstack((constructed_voxel,new_images))
        return constructed_voxel
    
class Dataset(torch_data.Dataset):
    def __init__(self, ids, targets, mri_type, if_pred = False):
        self.ids = ids
        self.targets = targets
        self.mri_type = mri_type
        self.if_pred = if_pred
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        scan_id = self.ids[index]
        data = construct_target_volume(scan_id,self.mri_type,scale_size=SCALE_SIZE)

        if self.if_pred:
            return {"X": torch.tensor(data).float().unsqueeze(0), "id":scan_id}
        else:
            y = torch.tensor(self.targets[index], dtype = torch.long)
            return {"X": torch.tensor(data).float().unsqueeze(0), "y": y}
        
# ============ Training ==============    
        
class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer,
        scheduler,
        lossfunction,
        mri_type
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfunction = lossfunction
        self.mri_type = mri_type

        self.best_valid_loss = np.inf
        self.n_patience = 0
        self.lastmodel = None

        # Save dirsctory used in saving model and records
        self.directory = f'{str(date.today())}-lr_{BASE_LR}-scale_size_{SCALE_SIZE}'
        
    def fit(self, epochs, train_loader, valid_loader, patience):
        epoch_all = []
        train_loss_all = []
        train_auc_all = []
        valid_loss_all = []
        valid_auc_all = [] 

        # ----------------------- Trainging -----------------------------
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_auc, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)

            # Record Training Process
            epoch_all.append(n_epoch)
            train_loss_all.append(train_loss)
            train_auc_all.append(train_auc)
            valid_loss_all.append(valid_loss)
            valid_auc_all.append(valid_auc)

            # Learning Rate Scheduler
            if self.scheduler:
                if self.scheduler.__module__ == lr_scheduler.__name__:
                    # Using PyTorch In-Built scheduler
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Current lr: {current_lr}')
                    self.scheduler.step(valid_loss)
                else:
                    # Using custom defined scheduler
                    updated_lr = self.scheduler(n_epoch-1)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = updated_lr
                    print(f'Current lr: {updated_lr}')

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, train_loss, train_auc, train_time
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

 
            if valid_loss < self.best_valid_loss: 
                self.save_model(n_epoch, valid_loss, valid_auc)
                self.info_message(
                     "Valid Loss improved from {:.4f} to {:.4f}. Saved model to '{}'", 
                    self.best_valid_loss, valid_loss, self.lastmodel
                )
                self.best_valid_loss = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1
            
            if self.n_patience >= patience:
                record_df = {'Epoch':epoch_all,'train loss':train_loss_all,'train AUC':train_auc_all,'valid loss':valid_loss_all,'valid AUC':valid_auc_all}
                record_df = pd.DataFrame(record_df)
                save_path = os.path.join(SAVE_FOLDER, self.directory)
                record_df.to_csv(f'{save_path}/{self.mri_type}/meta_classification.csv')
                self.info_message("\nValid Loss didn't improve last {} epochs.", patience)
                break
            
    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0
        y_all = []
        preds_all = []

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)

            loss = self.lossfunction(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()

            _, preds = torch.max(F.softmax(outputs,dim = 1), dim=1)
            y_all.extend(batch["y"].tolist())
            preds_all.extend(preds.tolist())
            
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")

        auc = roc_auc_score(y_all, preds_all)
        
        return sum_loss/len(train_loader), auc, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        preds_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X)

                loss = self.lossfunction(outputs, targets)

                sum_loss += loss.detach().item()

                _, preds = torch.max(F.softmax(outputs,dim = 1), dim=1)
                y_all.extend(batch["y"].tolist())
                preds_all.extend(preds.tolist())

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss/step, end="\r")
            
        auc = roc_auc_score(y_all, preds_all)
        
        return sum_loss/len(valid_loader), auc, int(time.time() - t)
    
    def save_model(self, n_epoch, loss, auc):
        today_date = date.today()
        save_path_parent = SAVE_FOLDER
        save_path = os.path.join(save_path_parent, self.directory)
        if self.directory not in os.listdir(save_path_parent):
            os.mkdir(save_path)

        save_path_modality = os.path.join(save_path, self.mri_type)
        if self.mri_type not in os.listdir(save_path):
            os.mkdir(save_path_modality)
        self.lastmodel = f"{save_path_modality}/{self.mri_type}-Epoch{n_epoch}-loss{loss:.3f}-auc{auc:.3f}.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_loss": self.best_valid_loss,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )
    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)

def train_U_Net(model,
                df_train, 
                df_valid, 
                mri_type,
                optimizer,
                scheduler,
                batch_size,
                n_workers,
                device,
                train_epoch: int = 20,
                load_saved_model = False
                ):
    
    train_data_retriever = Dataset(
        df_train["BraTS21ID"].values, 
        df_train["MGMT_value"].values,
        mri_type
    )

    valid_data_retriever = Dataset(
        df_valid["BraTS21ID"].values, 
        df_valid["MGMT_value"].values,
        mri_type
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory = True
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory = True
    )

    
    model.to(device)      

    lossfunction = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, 
        device, 
        optimizer,
        scheduler, 
        lossfunction,
        mri_type
    )

    history = trainer.fit(
        train_epoch, 
        train_loader, 
        valid_loader, 
        5
    )
    
    return trainer.lastmodel

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0,
                 warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                    math.pi *
                    (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

def train(epochs: int = 20,
          mri_type = 'flair',
          base_lr: float = 0.03,
          momentum: float = 0.9,
          weight_decay: float = 1e-3,
          use_scheduler = False,
          ):
    
    print(f"\n================= {NETWORK} Training MRI: {mri_type} ==================")
    model = Model()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum = momentum, weight_decay = weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr = base_lr,weight_decay = weight_decay)
    
    if use_scheduler:
        # scheduler = CosineScheduler(max_update = epochs-5, base_lr=base_lr, final_lr=1e-7)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=2, verbose=True)
    else:
        scheduler = None
    
    return train_U_Net(model,train_df,valid_df,mri_type, optimizer = optimizer, scheduler = scheduler, batch_size = args.batch_size,
                n_workers=args.n_workers,device = DEVICE,train_epoch = epochs)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': 2}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out
        
set_seed(42)

train_df,valid_df = load_and_split_labels(LABEL_PATH)
DEVICE = torch.device(f'cuda:{args.gpu}')



# ----------------Start Training------------------------
torch.cuda.empty_cache()
print(f'Pytorch Reserved Memory: {torch.cuda.memory_reserved()}')
for mri_type in MRI_TYPES:
    train(epochs = 30, mri_type =  mri_type, base_lr = args.base_lr, weight_decay = 0, use_scheduler = True) 