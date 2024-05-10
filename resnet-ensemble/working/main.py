from train import train_type_classifier_on_fold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

import torch
import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import config
from dataset import BrainRSNADataset

def train_pytorch_models():
    n_folds = 5
    scan_types = ['T1wCE', 'FLAIR', 'T1w', 'T2w']
    model_name = 'resnet10'
    scan_types = ['T1w', 'T2w']
    for scan in scan_types:
        for i in range(n_folds):
            print('#'*80)
            print(f'training: {scan} - {i}')
            train_type_classifier_on_fold(mri_type=scan, fold=i, model_name=model_name) #autosaves the best model, so don't need to store in a variable

def train_stacking_classifier():
    n_folds = 5
    device = torch.device("cuda")
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
    model.to(device)

    all_weights = os.listdir("../weights")
    data = pd.read_csv("../input/train.csv")
    targets = data['MGMT_value']

    all_pred_dicts = []
    all_averaged_preds = []
    all_y_true = []
    scan_types = ['T1wCE', 'FLAIR', 'T1w', 'T2w']
    
    for fold in range(n_folds):
        fold_preds = {'T1wCE':[], 'FLAIR':[], 'T1w':[], 'T2w':[]}
        val_df = data[data.fold == fold]
        val_index = val_df.index
        val_df = val_df.reset_index(drop=True)
        y_true_fold = targets[val_index].values
        for scan in scan_types:
            type_fold_files = [f for f in all_weights if scan+"_" in f]
            model.load_state_dict(torch.load(f"../weights/{type_fold_files[fold]}"))
            test_dataset = BrainRSNADataset(data=val_df, mri_type=scan, is_train=True, do_load=True, ds_type=f"val_{scan}_{fold}")
            test_dl = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
            image_ids = []

            preds = []
            case_ids = []
            with torch.no_grad():
                for  step, batch in enumerate(test_dl):
                    model.eval()
                    images = batch["image"].to(device)
                    outputs = model(images).sigmoid().detach().cpu().numpy()
                    fold_preds[f'{scan}'].extend(outputs)

        averaged_preds = []
        keys = list(fold_preds.keys())
        for i in range(len(fold_preds['T1wCE'])):
            pred = 0
            for key in keys:
                pred += fold_preds[key][i]
            pred /= len(keys)
            averaged_preds.append(pred)

        print(f'averaged performance on fold {fold}:')
        print(classification_report(y_true_fold, np.round(averaged_preds)))
        print('auc score:', roc_auc_score(y_true_fold, averaged_preds))
        print('accuracy:', accuracy_score(y_true_fold, np.round(averaged_preds)))
        all_averaged_preds.append(averaged_preds)
        all_pred_dicts.append(fold_preds)

def main():
    train_pytorch_models()
    train_stacking_classifier()

    
