import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from typing import Optional
from torch.nn import functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader


from transformers import AdamW, get_cosine_schedule_with_warmup

from get_dataloader import get_loader
from resnet_models import BasicNet
from focal_loss import FocalLoss
from cams.bagcams import BagCAMs, GradCAM
from Logger import Logger

import torchmetrics


def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor


def train_epoch(model:Optional[BasicNet or nn.Module], 
                device:torch.device, 
                train_loader:DataLoader, 
                optimizer:torch.optim.Optimizer, 
                shcheduler:torch.optim.lr_scheduler.LambdaLR,
                loss_fn:nn.modules.loss._Loss, 
                epoch:int,
                num_classes:int,
                log_interval=1):
    
    model.train()
    
    # Calculate the metric for each class separately, and return the metric for every class.
    acc_metric = torchmetrics.Accuracy(num_classes=num_classes, average=None).to(device)
    
    for batch_idx, ((images, masks, labels), image_ids) in enumerate(train_loader):
        
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        logits, probs = model(images)
        
        if isinstance(loss_fn, CrossEntropyLoss) or isinstance(loss_fn, FocalLoss):
            loss = loss_fn(logits, labels)
        else:
            loss = loss_fn(probs, labels)
            
        loss.backward()
        
        optimizer.step()
        
        acc = acc_metric(probs, labels)
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}] Loss: {:.6f}'.format(epoch, min(len(train_loader.dataset), (batch_idx+1) * train_loader.batch_size), len(train_loader.dataset), loss.item()))
    
    # lr schecule
    shcheduler.step()

    # return train accuracy
    acc = acc_metric.compute().cpu()
    acc_metric.reset()
    
    print('\nTrain Epoch: {} Acc: {}\n'.format(epoch, acc))
    
    return torch.mul(acc.sum(), 1/num_classes).item()


def val(model:Optional[BasicNet or nn.Module], 
        device:torch.device, 
        val_loader:DataLoader, 
        loss_fn:nn.modules.loss._Loss, 
        num_classes:int):
    
    model.eval()
    
    val_loss = 0
    
    # Calculate the metric for each class separately, and return the metric for every class.
    acc_metric = torchmetrics.Accuracy(num_classes=num_classes, average=None).to(device)
    recall_metric = torchmetrics.Recall(num_classes=num_classes, average=None).to(device)
    specificity_metric = torchmetrics.Specificity(num_classes=num_classes, average=None).to(device)
    
    with torch.no_grad():
        for batch_idx, ((images, masks, labels), image_ids) in enumerate(val_loader):
            
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            logits, probs = model(images)
            
            if isinstance(loss_fn, CrossEntropyLoss) or isinstance(loss_fn, FocalLoss):
                val_loss += loss_fn(logits, labels).item() * len(images)
            else:
                val_loss += loss_fn(probs, labels).item() * len(images)
            
            
            acc = acc_metric(probs, labels).cpu()
            recall = recall_metric(probs, labels).cpu()
            specificity = specificity_metric(probs, labels).cpu()
                 
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    acc = acc_metric.compute().cpu()
    recall = recall_metric.compute().cpu()
    specificity = specificity_metric.compute().cpu()
    print('\nVal set: Average loss: {:.4f}\n'.format(avg_val_loss))
    print('Val    Accuracy   on all data: {}'.format(acc))
    print('Val     Recall    on all data: {}'.format(recall))
    print('Val  Specificity  on all data: {}\n'.format(specificity))
    
    acc_metric.reset()
    recall_metric.reset()
    specificity_metric.reset()
    
    return torch.mul((acc+recall+specificity).sum(), 1/num_classes).item(), torch.mul(acc.sum(), 1/num_classes).item()
            

def test_and_generate_cams(model:Optional[BasicNet or nn.Module], 
                           device:torch.device, 
                           test_loader:DataLoader, 
                           num_classes:int, 
                           checkpoint_path:str, 
                           return_cams:bool=False,
                           target_layer:Optional[None or str]='layer3'):
    
    model.load_pretrained(checkpoint_path)
    model.eval()
    
    # Calculate the metric for each class separately, and return the metric for every class.
    acc_metric = torchmetrics.Accuracy(num_classes=num_classes, average=None).to(device)
    recall_metric = torchmetrics.Recall(num_classes=num_classes, average=None).to(device)
    specificity_metric = torchmetrics.Specificity(num_classes=num_classes, average=None).to(device)
    
    with torch.no_grad():
        for batch_idx, ((images, masks, labels), image_ids) in enumerate(test_loader):
            
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            logits, probs = model(images)
            
            acc = acc_metric(probs, labels).cpu()
            recall = recall_metric(probs, labels).cpu()
            specificity = specificity_metric(probs, labels).cpu()
    
    acc = acc_metric.compute().cpu()
    recall = recall_metric.compute().cpu()
    specificity = specificity_metric.compute().cpu()
    print('Test    Accuracy   on all data: {}, average: {:.4f}'.format(acc, acc.sum().item()/num_classes))
    print('Test     Recall    on all data: {}, average: {:.4f}'.format(recall, recall.sum().item()/num_classes))
    print('Test  Specificity  on all data: {}, average: {:.4f}\n'.format(specificity, specificity.sum().item()/num_classes))
    
    acc_metric.reset()
    recall_metric.reset()
    specificity_metric.reset()
    
    if return_cams:
        
        cams = None
        cam_computer = BagCAMs(extractor=model.extractor, classifier=model.classifier)
        # cam_computer = GradCAM(extractor=model.extractor, classifier=model.classifier)
        
        for batch_idx, ((images, masks, labels), image_ids) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            ###Grad-based CAMs
            logits, probs = cam_computer.forward(images)
            cam_computer.backward(ids=labels)
            batch_cams = cam_computer.generate(target_layer)
            ###
            
            '''
            ###Original CAMs
            pixel_features = model.extractor(images)
            output = model.classifier(pixel_features) # 0-1 mask
            image_size = images.shape[2:]
            output = F.interpolate(output, image_size, mode="bilinear", align_corners=False)
            batch_size = pixel_features.shape[0]
            normalized = normalize_tensor(output.detach().clone())
            batch_cams = normalized[range(batch_size), labels].unsqueeze(1)
            #print(batch_cams.shape, images.shape)
            #batch_cams = batch_cams.detach().cpu().numpy().astype(np.float)
            ###
            '''
            
            if cams is None:
                cams = torch.cat((images, masks, batch_cams), dim=1)
                # cams = batch_cams
            else:
                cams = torch.cat((cams, torch.cat((images, masks, batch_cams), dim=1)), dim=0)
                # cams = torch.cat((cams, batch_cams), dim=0)
            '''
            for i in range(test_loader.batch_size):
                result_path = os.path.join(cam_results_folder, image_ids+'.pt')
                result = torch.cat((images[i], masks[i], batch_cams[i]), dim=0).detach().cpu() # shape=(3, H, W)
                torch.save(result, result_path)
            '''
        return cams
    


def train_extractor(model:Optional[BasicNet or nn.Module], 
                device:torch.device, 
                train_loader:DataLoader,
                val_loader:DataLoader,
                test_loader:DataLoader,
                optimizer:torch.optim.Optimizer, 
                scheduler:torch.optim.lr_scheduler.LambdaLR,
                loss_fn:nn.modules.loss._Loss, 
                num_epochs:int,
                num_classes:int,
                experiment_tag:str,
                val_interval=2):
    # train extractor
    model = model.to(device)
    model.train()
    
    # save train epoch acc
    train_results = np.zeros([num_epochs, 2])
    
    # save validation results
    val_score = 0.
    val_results = np.zeros([int(num_epochs/val_interval), 3]) # epoch acc score
    saved_checkpoint_folder = os.path.join('saved_checkpoints', experiment_tag)
    
    
    epoch = 0
    while epoch < num_epochs:
        
        epoch += 1
        # train epoch
        train_results[epoch-1] = (epoch, train_epoch(model, device, train_loader, optimizer, scheduler, loss_fn, epoch, num_classes)) # record epoch acc
        
        # val model
        if epoch % val_interval == 0:
            score, acc = val(model, device, val_loader, loss_fn, num_classes)
            model.save_checkpoint(saved_checkpoint_folder, 'val-{}'.format(epoch))
            val_results[int(epoch/val_interval)-1] = (epoch, acc, score) # record epoch, acc, val average score: acc+recall+specificity
            # save the best model
            if score > val_score:
                model.save_checkpoint(saved_checkpoint_folder, 'val-best-{}'.format(epoch))
                val_score = score
    
    print('Train results:\n{}\n'.format(train_results))
    print('Val Results:\n{}\n'.format(val_results))
    
    # test model, do not return cams
    best_checkpoint = os.path.join(saved_checkpoint_folder, 'val-best-{}.pt'.format((val_results[:,1].argmax()+1)*val_interval))
    test_and_generate_cams(model, device, test_loader, num_classes, best_checkpoint, False)
    
    
    # record train curve and val curve
    fig, ax = plt.subplots()
    x = train_results[:, 0]
    y1 = train_results[:, 1] # train_acc
    y2 = val_results[:, 1] # val acc
    y3 = val_results[:, 2] # val_score
    
    ax.plot(x, y1, label='train-acc')
    ax.plot(x, y2, label='val-acc')
    
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    ax.set_title('experiment log')
    ax.legend()
    
    plt.savefig(os.path.join(saved_checkpoint_folder, 'experiment_log.png'))
    

if __name__ == '__main__':
    
    # general parameters
    num_classes = 3
    num_epochs = 300
    val_interval = 1
    lr = 0.9e-3
    
    # experiment tag
    experiment_tag = 'class3-celoss-9e-4_2'
    saved_checkpoint_folder = os.path.join('saved_checkpoints', experiment_tag)
    
    # set Logger
    sys.stdout = Logger(os.path.join(saved_checkpoint_folder, 'experiment_log.txt'))
    
    # set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    
    # set device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # set model
    model = BasicNet(1, num_classes, False).to(device)
    
    # set dataloader
    train_loader = get_loader('Data/Inexact_spines/Spine_train.txt', 'train', 0.5, 128, 16, 'Spine_Dataset')
    val_loader = get_loader('Data/Inexact_spines/Spine_val.txt', 'val', 0.5, 128, 16, 'Spine_Dataset')
    test_loader = get_loader('Data/Inexact_spines/Spine_test.txt', 'test', 0.5, 128, 16, 'Spine_Dataset')
    # train_loader = get_loader('Data/OLF_spines/OLF_train.txt', 'train', 0.5, 128, 16, 'OLF_Dataset')
    # val_loader = get_loader('Data/OLF_spines/OLF_val.txt', 'val', 0.5, 128, 16, 'OLF_Dataset')
    # test_loader = get_loader('Data/OLF_spines/OLF_test.txt', 'test', 0.5, 128, 16, 'OLF_Dataset')
        
    # set optimizer
    optimizer = AdamW(params=model.parameters(),
                      lr=lr,
                      betas=(0.9, 0.999),
                      eps=1e-6,
                      weight_decay=0.05)
    
    # set schedules
    warm_up_ratio = 0.2
    
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=num_epochs*warm_up_ratio,
                                                   num_training_steps=num_epochs)
    
    loss_fn = CrossEntropyLoss()
    # loss_fn = FocalLoss(gamma=2., alpha=0.25)
    
    print('lr: {}\nnum_epochs: {}\nseed: {}\nval_interval: {}\nwarm_up_ratio: {}'.format(lr, num_epochs, seed, val_interval, warm_up_ratio))
    
    train_extractor(model, device, train_loader, val_loader, test_loader, optimizer, lr_scheduler, loss_fn, num_epochs, num_classes, experiment_tag, val_interval)
    
    
