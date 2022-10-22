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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from skimage.transform import resize

import nibabel as nib
import matplotlib.pyplot as plt

from unet_down import UNet

import argparse

parser = argparse.ArgumentParser(description='U-Net Classification Train')
parser.add_argument('--mri_type', type=str,
                    help='Train your model on which MRI type. Should be one of: flair,t1,t1ce,t2. All (All means sequentially training the above 4 mri types)', default='flair')
parser.add_argument('--region', type=int,
                    help='Use which brain region to feed the model, Should be one of index:[1:Whole Brain, 2:Whole Tumor,3:WB+WT+Sub]', default=0)
parser.add_argument('--fold', type=int,
                    help='Use which fold to train the model range from 1-5', default=1)
parser.add_argument('--gpu', type=int,
                    help='GPU ID', default=0)
parser.add_argument('--batch_size', type=int,
                    help='Batch size', default=2)
parser.add_argument('--base_lr', type=float,
                    help='Base Learning Rate', default=1e-5)
parser.add_argument('--weight_decay', type=float,
                    help='Weight Decay', default=1e-12)
parser.add_argument('--scale_size', type=int,
                    help='Volume Size', default=90)
parser.add_argument('--n_workers', type=int,
                    help='Number of parrallel workers', default=8)
args = parser.parse_args()

RAW_DATA_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_Prep_Segmentation'
MASKS_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_Masks_Train'
Rescale_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_WB_96_96_62'

LABEL_PATH = './train_labels.csv'
BASE_LR = args.base_lr
IMAGE_SIZE = 90
SCALE_SIZE = args.scale_size
INPUT_SIZE = (IMAGE_SIZE,IMAGE_SIZE,SCALE_SIZE)
FOLDS = range(args.fold)
SEED = 42

print(f'Input Size:{INPUT_SIZE}')
SAVE_FOLDER = './Model'

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

def get_train_valid_split(label_path):
    train_df = pd.read_csv(label_path,dtype = {'BraTS21ID':'str','MGMT_value':'int'})
    index_name = train_df[(train_df['BraTS21ID'] == '00109') | (train_df['BraTS21ID'] == '00123') | (train_df['BraTS21ID'] == '00709')].index
    train_df = train_df.drop(index_name).reset_index(drop=True)

    X = train_df['BraTS21ID'].values
    y = train_df['MGMT_value'].values
    
    kfold =  StratifiedKFold(n_splits=5,shuffle = True,random_state = SEED)
    return X,y,list(kfold.split(X,y))


def non_0_voxel(voxel):
    length = voxel.shape[2]
    start_id = 0
    end_id = length-1

    # From begining to find start index
    for i in range(length):
        if np.max(voxel[:,:,i]) != 0:
            start_id = i
            break

    # From final to find end index
    for i in range(length-1,-1,-1):
        if np.max(voxel[:,:,i]) != 0:
            end_id = i
            break
    non_0_indexs = slice(start_id,end_id+1)
    
    return non_0_indexs 


WB_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_WB_90_90_90'
WT_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_WT_90_90_90'
TC_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_TC_90_90_90'
ET_PATH = '/mnt/24CC5B14CC5ADF9A/Brain_Tumor_Classification/Datasets/Data_ET_90_90_90'
"""
# ---------------------- WB/WT 130 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=130):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WT = nib.load(f'{WT_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WB[voxel_WB < 1e-1 ] = 0
    voxel_WT[voxel_WT < 1e-1] = 0
    
        
    non_0_slice = non_0_voxel(voxel_WB)
    
    non_0_WB = voxel_WB[:,:,non_0_slice]
    non_0_WT = voxel_WT[:,:,non_0_slice]
    
    half_length = scale_size//2
    diff = half_length-non_0_WB.shape[2]
    if diff<0:
        diff = abs(diff)
        div, mod = divmod(diff,2)
        start = div+mod-1
        end = non_0_WB.shape[2]-div-1

        constructed_voxel = np.dstack((non_0_WB[:,:,slice(start,end)],non_0_WT[:,:,slice(start,end)]))
    else:
        div, mod = divmod(diff,2)
        before = div+mod
        after = div

        fixed_WB = np.pad(non_0_WB,((0,0),(0,0),(before,after)))
        fixed_WT = np.pad(non_0_WT,((0,0),(0,0),(before,after)))

        constructed_voxel = np.dstack((fixed_WB,fixed_WT))
    return constructed_voxel

"""
"""
# ---------------------- WB/WT/ET 195 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WT = nib.load(f'{WT_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_ET = nib.load(f'{ET_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    
    voxel_WB[voxel_WB < 1e-1 ] = 0
    voxel_WT[voxel_WT < 1e-1] = 0
    voxel_ET[voxel_ET < 1e-1 ] = 0
    
    non_0_slice = non_0_voxel(voxel_WB)
    
    non_0_WB = voxel_WB[:,:,non_0_slice]
    non_0_WT = voxel_WT[:,:,non_0_slice]
    non_0_ET = voxel_ET[:,:,non_0_slice]
    
    split_length = scale_size//3
    diff = split_length-non_0_WB.shape[2]
    if diff<0:
        diff = abs(diff)
        div, mod = divmod(diff,2)
        start = div+mod-1
        end = non_0_WB.shape[2]-div-1

        constructed_voxel = np.dstack((non_0_WB[:,:,slice(start,end)],non_0_WT[:,:,slice(start,end)],non_0_ET[:,:,slice(start,end)]))
    else:
        div, mod = divmod(diff,2)
        before = div+mod
        after = div

        fixed_WB = np.pad(non_0_WB,((0,0),(0,0),(before,after)))
        fixed_WT = np.pad(non_0_WT,((0,0),(0,0),(before,after)))
        fixed_ET = np.pad(non_0_ET,((0,0),(0,0),(before,after)))

        constructed_voxel = np.dstack((fixed_WB,fixed_WT,fixed_ET))
    return constructed_voxel
"""
"""
# ---------------------- WT/TC/ET 195 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WT = nib.load(f'{WT_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_TC = nib.load(f'{TC_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_ET = nib.load(f'{ET_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    
    voxel_WB[voxel_WB < 1e-1 ] = 0
    voxel_WT[voxel_WT < 1e-1] = 0
    voxel_TC[voxel_TC < 1e-1] = 0
    voxel_ET[voxel_ET < 1e-1 ] = 0
    
    non_0_slice = non_0_voxel(voxel_WB)
    
    non_0_WB = voxel_WB[:,:,non_0_slice]
    non_0_WT = voxel_WT[:,:,non_0_slice]
    non_0_TC = voxel_TC[:,:,non_0_slice]
    non_0_ET = voxel_ET[:,:,non_0_slice]
    
    split_length = scale_size//3
    diff = split_length-non_0_WB.shape[2]
    if diff<0:
        diff = abs(diff)
        div, mod = divmod(diff,2)
        start = div+mod-1
        end = non_0_WB.shape[2]-div-1

        constructed_voxel = np.dstack((non_0_WT[:,:,slice(start,end)],non_0_TC[:,:,slice(start,end)],non_0_ET[:,:,slice(start,end)]))
    else:
        div, mod = divmod(diff,2)
        before = div+mod
        after = div

        fixed_WB = np.pad(non_0_WB,((0,0),(0,0),(before,after)))
        fixed_WT = np.pad(non_0_WT,((0,0),(0,0),(before,after)))
        fixed_TC = np.pad(non_0_TC,((0,0),(0,0),(before,after)))
        fixed_ET = np.pad(non_0_ET,((0,0),(0,0),(before,after)))

        constructed_voxel = np.dstack((fixed_WT,fixed_TC,fixed_ET))
    return constructed_voxel
"""
"""
# ---------------------- WB/WT/ET/TC 260 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WT = nib.load(f'{WT_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_TC = nib.load(f'{TC_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_ET = nib.load(f'{ET_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    
    voxel_WB[voxel_WB < 1e-1 ] = 0
    voxel_WT[voxel_WT < 1e-1] = 0
    voxel_TC[voxel_TC < 1e-1] = 0
    voxel_ET[voxel_ET < 1e-1 ] = 0
    
    non_0_slice = non_0_voxel(voxel_WB)
    
    non_0_WB = voxel_WB[:,:,non_0_slice]
    non_0_WT = voxel_WT[:,:,non_0_slice]
    non_0_TC = voxel_TC[:,:,non_0_slice]
    non_0_ET = voxel_ET[:,:,non_0_slice]
    
    split_length = scale_size//4
    diff = split_length-non_0_WB.shape[2]
    if diff<0:
        diff = abs(diff)
        div, mod = divmod(diff,2)
        start = div+mod-1
        end = non_0_WB.shape[2]-div-1

        constructed_voxel = np.dstack((non_0_WB[:,:,slice(start,end)],non_0_WT[:,:,slice(start,end)],non_0_ET[:,:,slice(start,end)],non_0_TC[:,:,slice(start,end)]))
    else:
        div, mod = divmod(diff,2)
        before = div+mod
        after = div

        fixed_WB = np.pad(non_0_WB,((0,0),(0,0),(before,after)))
        fixed_WT = np.pad(non_0_WT,((0,0),(0,0),(before,after)))
        fixed_TC = np.pad(non_0_TC,((0,0),(0,0),(before,after)))
        fixed_ET = np.pad(non_0_ET,((0,0),(0,0),(before,after)))

        constructed_voxel = np.dstack((fixed_WB,fixed_WT,fixed_ET,fixed_TC))
    return constructed_voxel
"""
"""
# ---------------------- WB/WT/TC/ET 260 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_WT = nib.load(f'{WT_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_TC = nib.load(f'{TC_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    voxel_ET = nib.load(f'{ET_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    
    voxel_WB[voxel_WB < 1e-1 ] = 0
    voxel_WT[voxel_WT < 1e-1] = 0
    voxel_TC[voxel_TC < 1e-1] = 0
    voxel_ET[voxel_ET < 1e-1 ] = 0
    
    non_0_slice = non_0_voxel(voxel_WB)
    
    non_0_WB = voxel_WB[:,:,non_0_slice]
    non_0_WT = voxel_WT[:,:,non_0_slice]
    non_0_TC = voxel_TC[:,:,non_0_slice]
    non_0_ET = voxel_ET[:,:,non_0_slice]
    
    split_length = scale_size//4
    diff = split_length-non_0_WB.shape[2]
    if diff<0:
        diff = abs(diff)
        div, mod = divmod(diff,2)
        start = div+mod-1
        end = non_0_WB.shape[2]-div-1

        constructed_voxel = np.dstack((non_0_WB[:,:,slice(start,end)],non_0_WT[:,:,slice(start,end)],non_0_TC[:,:,slice(start,end)],non_0_ET[:,:,slice(start,end)]))
    else:
        div, mod = divmod(diff,2)
        before = div+mod
        after = div

        fixed_WB = np.pad(non_0_WB,((0,0),(0,0),(before,after)))
        fixed_WT = np.pad(non_0_WT,((0,0),(0,0),(before,after)))
        fixed_TC = np.pad(non_0_TC,((0,0),(0,0),(before,after)))
        fixed_ET = np.pad(non_0_ET,((0,0),(0,0),(before,after)))

        constructed_voxel = np.dstack((fixed_WB,fixed_WT,fixed_TC,fixed_ET))
    return constructed_voxel
"""

"""
# ---------------------- WB 96 96 62 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_WB = nib.load(f'{WB_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    return voxel_WB
"""
"""
# ---------------------- TC 90 90 90 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_TC = nib.load(f'{TC_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    return voxel_TC
"""

# ---------------------- ET 90 90 90 ---------------------------
def construct_target_volume(scan_id,mri_type,scale_size=260):
    voxel_ET = nib.load(f'{ET_PATH}/BraTS2021_{scan_id}/BraTS2021_{scan_id}_{mri_type}.nii.gz').get_fdata().astype('float')
    return voxel_ET


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
        mri_type,
        fold
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfunction = lossfunction
        self.mri_type = mri_type
        self.fold = fold

        self.best_valid_loss = np.inf
        self.n_patience = 0
        self.lastmodel = None

        # Save dirsctory used in saving model and records
        self.directory = f'{str(date.today())}-base_lr{BASE_LR}-image_size{IMAGE_SIZE}-scale_size{SCALE_SIZE}'
        
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
                record_df.to_csv(f'{save_path}/{self.mri_type}/meta_classification_fold{self.fold}.csv')
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

            # _, preds = torch.max(F.softmax(outputs,dim = 1), dim=1)
            preds = outputs.softmax(dim=1)[:,1]
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

                # _, preds = torch.max(F.softmax(outputs,dim = 1), dim=1)
                preds = outputs.softmax(dim=1)[:,1]
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
        self.lastmodel = f"{save_path_modality}/{self.mri_type}-fold{self.fold}-best.pth"
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
                X_train, 
                X_valid,
                y_train, 
                y_valid,
                mri_type,
                optimizer,
                scheduler,
                batch_size,
                n_workers,
                device,
                fold,
                train_epoch: int = 20,
                load_saved_model = False
                ):
    
    train_data_retriever = Dataset(
        X_train, 
        y_train,
        mri_type
    )

    valid_data_retriever = Dataset(
        X_valid, 
        y_valid,
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
        mri_type,
        fold
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
        
    
    for fold in FOLDS:
        model = UNet(in_channels=1,
                 out_channels=2,
                 n_blocks=4,
                 input_shape = INPUT_SIZE,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=3,
                 hidden_channels=2048)
    
        # optimizer = torch.optim.SGD(model.parameters(), lr = base_lr, momentum = momentum, weight_decay = weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr = base_lr,weight_decay = weight_decay)
    
        if use_scheduler:
            # scheduler = CosineScheduler(max_update = epochs-5, base_lr=base_lr, final_lr=1e-7)
            scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=2, verbose=True)
        else:
            scheduler = None
            
        train_idx,valid_idx = SPLIT[fold]
        X_train,X_valid = X[train_idx],X[valid_idx]
        y_train,y_valid = y[train_idx],y[valid_idx]
        
        print(f"\n================= U-Net Training MRI: {mri_type} Fold: {fold} ==================")
        train_U_Net(model,X_train,X_valid,y_train,y_valid,mri_type, optimizer = optimizer, scheduler = scheduler, batch_size = args.batch_size,
                n_workers=args.n_workers,device = DEVICE,fold = fold,train_epoch = epochs)
        
        del model


        
set_seed(SEED)
DEVICE = torch.device(f'cuda:{args.gpu}')
X,y,SPLIT = get_train_valid_split(LABEL_PATH)
print(f'Number of training samples: {len(SPLIT[0][0])}. Number of valid samples: {len(SPLIT[0][1])}')

# ----------------Start Training------------------------
torch.cuda.empty_cache()
print(f'Pytorch Reserved Memory: {torch.cuda.memory_reserved()}')

for mri_type in MRI_TYPES:
    train(epochs = 50, mri_type =  mri_type, base_lr = args.base_lr, weight_decay = args.weight_decay, use_scheduler = True) 