#import libraries
import os
import glob
import datetime
import numpy as np
import pandas as pd
import rioxarray
from PIL import Image
from collections import Counter
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.optim import Adam
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from planet_img_utils import find_day
from matplotlib import pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.image_path.values[idx]
        img = rioxarray.open_rasterio(img_path)
        img = img.data[0:3].transpose(1, 2, 0)
        label = self.df.label.values[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# TODO: Experiment with different transforms later
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

#TODO: problems with the train_test_split
#TODO: problems with the training loop, only one input is passed to the model

def train(data_frame,train_transform, test_transform, test_size = 0.1, num_epochs = 10, save_path = './models/', batch_size = 1):
    train, test = train_test_split(data_frame, test_size=test_size, random_state=42, stratify=data_frame.label.values)
    test.reset_index(inplace=True, drop=True)
    train.reset_index(inplace=True, drop=True)
    print("Train size:", train.shape)
    print("Test size:", test.shape)
    print("Train class distribution:", train.label.value_counts())
    print("Test class distribution:", test.label.value_counts())



    train_dataset = CustomDataset(train, transform=train_transform)
    test_dataset = CustomDataset(test, transform=test_transform)

    if batch_size == 1:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=my_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=my_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(512, 2)
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    train_auc = []
    test_auc = []

    best_accuracy = 0.0
    best_AUC = 0.0
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_auc = 0.0
        y_pred = []
        y_true = []

        for inputs, labels in train_loader:
            labels = labels.to(device)

            if batch_size == 1:
                inputs = inputs[0].to(device)
                inputs = inputs.unsqueeze(0)
            else:
                inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # running_auc += roc_auc_score(labels.data.cpu(), outputs.data.cpu()[:,1])
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        epoch_auc = roc_auc_score(y_true, y_pred)
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc.item())
        train_auc.append(epoch_auc)
        print('Train Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, epoch_auc))

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        y_pred = []
        y_true = []
        for inputs, labels in test_loader:
            labels = labels.to(device)

            if batch_size == 1:
                inputs = inputs[0].to(device)
                inputs = inputs.unsqueeze(0)
            else:
                inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects.double() / len(test_dataset)
        epoch_auc = roc_auc_score(y_true, y_pred)
        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc.item())
        test_auc.append(epoch_auc)
        print('Test Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, epoch_auc))


        if save_path is not None:
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                print(("Saving the best model with accuracy {:.4f}").format(best_accuracy))
                torch.save(model.state_dict(), f"{save_path}_acc.pth")
            if epoch_auc > best_AUC:
                best_AUC = epoch_auc
                print(("Saving the best model with AUC {:.4f}").format(best_AUC))
                torch.save(model.state_dict(), f"{save_path}_auc.pth")

    return model, train_loss, train_accuracy, train_auc, test_loss, test_accuracy, test_auc

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def plot_gcam(model, path2img, test_transform, mean, std, plot = True):
    img = rioxarray.open_rasterio(path2img)
    img = img.data[0:3].transpose(1, 2, 0)
    input_tensor = test_transform(img)
    input_tensor = input_tensor.to(device)
    target_layers = [model.layer4[-1]] #TODO: Experiment with different layers
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = 1-grayscale_cam
    inp = std * input_tensor.cpu().numpy().transpose((1, 2, 0)) + mean
    rgb_img = np.clip(inp, 0, 1)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    output = model(input_tensor.unsqueeze(0))
    pred = torch.argmax(output, dim=1).cpu().item()
    print("Predicted label:", pred)
    date_in = path2img.split('/')[-1][:8]
    year = date_in[0:4]
    month = date_in[4:6]
    day = date_in[6:8]
    date_string = f"{year}-{month}-{day}"
    print("Day:", find_day(date_string))
    if find_day(date_string) == 'Sunday':
        print("True label: 1")
    else:
        print("True label: 0")

    #plot original image and gradcam image
    if plot:
        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(visualization)

    return img, visualization