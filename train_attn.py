import numpy as np
import os
from data import PAIRDataset, ClfDataset
from models import MAX_model, SUM_model, ContrastiveLoss, Siamese_Attn
from torch import nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.models import resnet50
from sklearn.metrics import classification_report
from torchvision.ops import sigmoid_focal_loss as criterion
import random
import matplotlib.pyplot as plt
from train import read_data
from calc_metrics2 import calc_froc, get_error_boxes, save_cases, join_cases, calc_accuracy



def create_plot(data, exp_name):
    # Plot the losses
    loss_names = ['0.05', '0.1', '0.15', '0.2', '0.3', '0.5']
    epochs = list(data.keys())
    epochs = sorted(epochs)
    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.grid(True)

    for i, loss_name in enumerate(loss_names):
        loss_values = [data[epoch][i] for epoch in epochs]
        plt.plot(epochs, loss_values, label=loss_name)

    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity')
    plt.title('Metrics')
    plt.legend()
    plt.savefig('{}/metric_plot.png'.format(exp_name))
    return




def change_confs(folder_path, mlo_scores, cc_scores):
    # import pdb; pdb.set_trace()
    num_props = mlo_scores.shape[0]
    folder_data = read_data(folder_path[0])
    if(folder_data[0]["view"]=="CC"):
        folder_data.append(folder_data.pop(0))

    folder_data[0]["pred"]["scores"] = folder_data[0]["pred"]["scores"][:num_props]
    folder_data[1]["pred"]["scores"] = folder_data[1]["pred"]["scores"][:num_props]
    folder_data[0]["pred"]["boxes"] = folder_data[0]["pred"]["boxes"][:num_props]
    folder_data[1]["pred"]["boxes"] = folder_data[1]["pred"]["boxes"][:num_props]
    folder_data[0]["pred"]["labels"] = folder_data[0]["pred"]["labels"][:num_props]
    folder_data[1]["pred"]["labels"] = folder_data[1]["pred"]["labels"][:num_props]

    # import pdb; pdb.set_trace()
    # folder_data[0]["pred"]["new_scores"] = mlo_scores
    # folder_data[1]["pred"]["new_scores"] = cc_scores
    folder_data[0]["pred"]["scores"] = mlo_scores
    folder_data[1]["pred"]["scores"] = cc_scores
    
    return folder_data


def test(data_file_path, model_path, wsm=False, resnet_path=""):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    sigmoid = nn.Sigmoid()
    dataset = PAIRDataset(pairs_path=data_file_path, transform=transform)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16)
    if(wsm):
        model = SUM_model(weights = None).to("cuda")
    else:
        model = MAX_model(weights = None).to("cuda")
            
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        pred_list = []
        # import pdb; pdb.set_trace()
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")
            
            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]
            # import pdb; pdb.set_trace()

            preds_mlo = model(mlo_data, cc_data)
            preds_mlo = sigmoid(preds_mlo)
            targets_mlo = mlo_data[1][:,5]

            preds_cc = model(cc_data, mlo_data)
            preds_cc = sigmoid(preds_cc)
            targets_cc= cc_data[1][:,5]

            pred_list+=change_confs(mlo_data[2], preds_mlo.cpu(), preds_cc.cpu())
            # pred_list = None
        fps_req, senses_req, thresh = calc_froc(pred_list)
        # fpi, sens = calc_froc(pred_list)
        fpi, sens = [], []
        tpr, fpr, precs = calc_accuracy(pred_list)
        
    return  tpr, fpr, fpi, sens, precs, pred_list

def val(dataloader, model, cosine_model=False):
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        pred_list = []
        loss_item = 0
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")

            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            # cc_data[1][:,4] = torch.softmax(cc_data[1][:,4], dim=0)
            # mlo_data[1][:,4] = torch.softmax(mlo_data[1][:,4], dim=0)

            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss_fn(preds, targets)

            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            loss_item += (loss/(mlo_data[0].shape[0]*cc_data[0].shape[0])).item()
        return loss_item


def get_dataloaders(data_file_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = PAIRDataset(pairs_path=data_file_path, transform=transform)
    batch_size = 1
    
    num_samples = len(dataset)
    split_ratio = 0.9
    split_idx = int(split_ratio * num_samples)
    indices = torch.randperm(num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=16)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=16)

    return train_loader, val_loader


def train(data_path, args, exp_name, wsm=False):
    epochs_, lr_, _, resnet_path = args
    train_loader, val_loader = get_dataloaders(data_path)
    if(wsm):
        model = SUM_model(weights = resnet_path).to("cuda")
    else:
        model = MAX_model(weights = resnet_path).to("cuda")
    
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer = optim.Adam(model.parameters(), lr=lr_)
    best_loss = 9999
    train_loss = []
    val_loss = []
    epoch_vals = []

    for i in range(epochs_):
        loss_item = 0
        loss = 0
        epoch_vals.append(i+1)
        for j, (mlo_data, cc_data) in enumerate(tqdm(train_loader)):
            mlo_data[0] = mlo_data[0].squeeze(0).to("cuda")
            mlo_data[1] = mlo_data[1].squeeze(0).to("cuda")
            cc_data[0] = cc_data[0].squeeze(0).to("cuda")
            cc_data[1] = cc_data[1].squeeze(0).to("cuda")

            max_props = min(min(mlo_data[0].shape[0],cc_data[0].shape[0]),25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            loss = 0
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:,5]
            preds = sigmoid(preds)
            loss  = loss + loss_fn(preds, targets)

            loss = loss/(mlo_data[0].shape[0]*cc_data[0].shape[0])
            loss.backward()
            optimizer.step()            
            loss_item+=loss.item()
        
        val_loss_item = val(val_loader, model)
        print("Epoch:", "{}/{}".format(i+1, epochs_), f"Train Loss: {loss_item:.4f}", f"Val Loss: {val_loss_item:.4f}")

        train_loss.append(loss_item)
        val_loss.append(val_loss_item)

        if(val_loss_item<best_loss):
            best_loss = val_loss_item
            PATH = "{}/{}_epoch.pth".format(exp_name, i)
            torch.save(model.state_dict(), PATH)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    # Plotting training loss
    ax1.plot(epoch_vals, train_loss, 'b-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    # Plotting validation loss
    ax2.plot(epoch_vals, val_loss, 'r-', label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()

    # Adjusting the spacing between subplots
    plt.subplots_adjust(hspace=0.4)

    # Display the plot
    plt.savefig('{}/loss_plot.png'.format(exp_name))


def save_plot_values(model_name, dataset, data):
    tpr, fpr, fpi, sen, precs = data
    # import pdb; pdb.set_trace()
    
    # auc_target_path = os.path.join("AUC","{}_{}_auc".format(dataset, model_name))
    # np.save(auc_target_path, np.array([fpr,tpr]))
    
    # froc_target_path = os.path.join("FROC","{}_{}_froc".format(dataset, model_name))
    # np.save(froc_target_path, np.array([fpi,sens]))
 
    pr_target_path = os.path.join("PR","{}_{}_pr".format(dataset, model_name))
    np.save(pr_target_path, np.array([precs,tpr]))
    


if __name__ == '__main__':
    dataset_name = ""
    train_data_path = ""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
 
    lr = 0.00001
    resnet_path = "resnet_pretrained_focal_{}.pth".format(dataset_name)
    args = [100, lr, 1, resnet_path]
    wsm = False
    if(wsm):
        exp_name = "./expts_weights/wsm_exps/{}_{}_resnet/wsm_{}".format(dataset_name, resolution, lr)
    else:
        exp_name = "./expts_weights/max_exps/{}_{}_resnet/max_{}".format(dataset_name, resolution, lr)
    # os.makedirs(exp_name,exist_ok=True)
    if(not os.path.isdir(exp_name)):
        print("PATH DOES NOT EXISTS", exp_name)
        exit(0)
    
    train(train_data_path, test_data_path, args, exp_name, wsm = wsm)
    
    # metrics = {}
    # model_paths = glob.glob(exp_name+"/9*.pth")
    # list.sort(model_paths)aii
    # print(model_paths)
    # for i,model_path in enumerate(model_paths):
    #     epoch_number = int(model_path.split("/")[-1].split("_")[0])
    #     print(model_path, test_data_path)
    #     sensitivity, _, _ = test(test_data_path, model_path, wsm = wsm, resnet_path=resnet_path)
    #     metrics[epoch_number] = sensitivity[1:7]
    
    # create_plot(metrics, exp_name)


