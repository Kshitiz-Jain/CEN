import numpy as np
import os
from data import PAIRDataset, ClfDataset
from models import SiameseNetwork
from torch import nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision.models import resnet50
from sklearn.metrics import classification_report
from torchvision.ops import sigmoid_focal_loss as criterion
import random
from sklearn.model_selection import train_test_split

def get_dataloaders(data_file_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset2 = ClfDataset(pairs_path=data_file_path, transform=transform)    
    batch_size = 128
    train_indices, valid_indices = train_test_split( list(range(len(dataset2))), test_size=0.2, stratify=dataset2.targets)
    
    # Calculate class weights for weighted sampling on the training set
    train_targets = [dataset2.targets[idx] for idx in train_indices]
    class_sample_counts = [train_targets.count(class_idx) for class_idx in range(2)]
    # import pdb;pdb.set_trace()
    class_weights = 1.0 / np.array(class_sample_counts, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float64)
    train_targets[np.where(np.array(train_targets)==0)[0]] = class_weights[0]
    train_targets[np.where(np.array(train_targets)==1)[0]] = class_weights[1]
    # Define sampler for weighted sampling on the training set
    train_targets = torch.tensor(train_targets / np.sum(train_targets))  # Normalize to sum to 1
    
    train_targets = train_targets.to("cuda")
    sampler = WeightedRandomSampler(train_targets, train_targets.shape[0]*2, replacement=True)

    train_loader = DataLoader(dataset2, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(dataset2, batch_size=batch_size, sampler=valid_indices, num_workers=4)

    return train_loader, val_loader


def test_resnet(data_path, args, focal = False):
    _,_,_,dataset_name = args
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ClfDataset(pairs_path=data_path, transform=transform)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16)
    
    model = resnet50(pretrained=True)
    num_classes = 2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    if(focal):
        model.load_state_dict(torch.load("resnet_pretrained_focal_{}.pth".format(dataset_name)))
    else:
        model.load_state_dict(torch.load("resnet_pretrained_{}.pth".format(dataset_name)))
    model = model.to("cuda")
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            targets+=labels.cpu().numpy().tolist()
            preds += predicted.cpu().numpy().tolist()
    class_report_val = classification_report(targets, preds)
    print(class_report_val)
    # val_f1 = class_report_val["1"]["f1-score"]
    

def train_resnet(data_path, args, focal=False):
    train_loader, val_loader = get_dataloaders(data_path)
    epochs_, lr_, _, dataset_name = args
    model = resnet50(pretrained=True)
    num_classes = 2
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to("cuda")

    # Define loss function and optimizer
    if not focal: criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_)

    # Training loop
    best_f1 = 0
    for epoch in range(epochs_):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        targets = []
        preds = []
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(images)
            if(focal):
                labels_hot = torch.nn.functional.one_hot(labels).type(torch.DoubleTensor).to("cuda")
                loss = criterion(outputs, labels_hot)
                loss = loss.mean()
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            targets+=labels.cpu().numpy().tolist()
            preds += predicted.cpu().numpy().tolist()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct / total
        class_report = classification_report(targets, preds, output_dict = True)
        train_f1 = class_report["1"]["f1-score"]

        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        targets = []
        preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to("cuda"), labels.to("cuda")

                outputs = model(images)
                if(focal):
                    labels_hot = torch.nn.functional.one_hot(labels).type(torch.DoubleTensor).to("cuda")
                    loss = criterion(outputs, labels_hot)
                    loss = loss.mean()
                else:
                    loss = criterion(outputs, labels)
                    
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                targets+=labels.cpu().numpy().tolist()
                preds += predicted.cpu().numpy().tolist()
        class_report_val = classification_report(targets, preds, output_dict = True)
        val_f1 = class_report_val["1"]["f1-score"]
        if(val_f1>best_f1):
            best_f1=val_f1
            if(focal): 
                torch.save(model.state_dict(), "resnet_pretrained_focal_{}.pth".format(dataset_name))
            else:
                torch.save(model.state_dict(), "resnet_pretrained_{}.pth".format(dataset_name))

        valid_loss /= len(val_loader)
        valid_accuracy = 100 * correct / total

        # Print the metrics for each epoch
        print(f"Epoch {epoch+1}/{epochs_} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.2f}%, "
            f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%, Valid F1: {val_f1:.2f}%")




if __name__ == '__main__':
    dataset_name = ""
    resolution = "4k"
    train_data_path = ""
    test_data_path= ""
    ROOT =  ""
    model_path = os.path.join(ROOT, "7_true_lr5_train.pth") 
    #epochs, lr, scheduler
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42) 
    
    args = [20, 0.00001, 50, datast_name]
    train_resnet(train_data_path, args, focal=True)
    test_resnet(test_data_path, args, focal=True)
    # pred_list = test(test_data_path, model_path)
    # import pdb; pdb.set_trace()
    # torch.save(pred_list, os.path.join(ROOT,"val_preds_vanilla_{}_{}.dict".format(7,0.5))) 
    # print()
    # pred_list = torch.load(os.path.join(ROOT,"test_preds_vanilla_{}_{}.dict".format(7,0.5)))
    # # calc_froc(pred_list)
 
    # pred_list = torch.load(os.path.join(ROOT,"test_preds_siamese_{}_{}.dict".format(7,0.5))) 
    # # calc_froc(pred_list)
    # train(test_data_path, args)