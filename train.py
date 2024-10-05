from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from loguru import logger 
import librosa
from tqdm import tqdm
from datetime import datetime
import os
import json
import wandb
import shutil
from pathlib import Path

from dataset.timit import TimitDataset, SAMPLE_RATE

class FrameLevelPhonemeClassifier(nn.Module):
    def __init__(self, num_labels, layer_list=[512]):
        super(FrameLevelPhonemeClassifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        
        # self.linear = nn.Linear(768, num_labels)
        layers = []
        for l_idx in range(len(layer_list)):
            if l_idx == 0:
                layers.append(nn.Linear(512, layer_list[l_idx]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(layer_list[l_idx-1], layer_list[l_idx]))
                layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.head = nn.Linear(layer_list[-1], num_labels)
    
    def forward(self, input_values, attention_mask=None):
        # outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        # hidden_states = outputs.last_hidden_state
        # logits = self.linear(hidden_states)
        # return logits

        outputs = self.wav2vec2.feature_extractor(input_values=input_values)
        outputs = self.hidden_layers(outputs[:, :, 0])
        logits = self.head(outputs)
        return logits

def train(model, data_loader, criterion, optimizer, epoch, total_epochs, device):
    log_interval = 50
    log_total_loss = 0
    total_correct = 0
    total_samples = 0

    model.train()

    for itr, batch in enumerate(data_loader):
        inputs, labels = batch
        labels = labels.to(device)

        # input_values = processor(inputs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)['input_values'].squeeze(0)
        # input_values = input_values.to(device)
        # outputs = model(input_values)
        # outputs = outputs[:, 0, :]

        input_values = inputs.to(device)
        outputs = model(input_values)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.shape[0]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        log_total_loss += loss.item()
        if itr != 0 and itr % log_interval == 0:
            accuracy = 100 * total_correct / total_samples
            loss_value = log_total_loss / log_interval
            lr = optimizer.param_groups[0]["lr"] 
            logger.info(f'[train] epoch {epoch:03d}/{total_epochs:03d}, iter {itr:04d}/{len(data_loader)}, loss: {loss_value:.3f}, acc: {accuracy:.2f}%, lr: {lr}')
            log_total_loss = 0
            total_correct = 0
            total_samples = 0
            wandb.log({'epoch': epoch, 'total_iter': epoch*len(data_loader)+itr, 'train-loss': loss_value, 'train-accuracy': accuracy, 'lr': lr})
    
def val(model, data_loader, criterion, epoch, total_epochs, device):
    total_correct = 0
    total_samples = 0
    total_loss = 0

    model.eval()

    for itr, batch in enumerate(tqdm(data_loader, desc='Validation')):
        inputs, labels = batch
        labels = labels.to(device)

        input_values = inputs.to(device)
        outputs = model(input_values)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.shape[0]
        total_loss += loss.item()

    accuracy = 100 * total_correct / total_samples
    logger.info(f'[validation] epoch {epoch:03d}/{total_epochs:03d}, loss: {total_loss / total_samples:.3f}, acc: {accuracy:.2f}%')
    return loss, accuracy

def test(model, data_loader, criterion, device):
    total_correct = 0
    total_samples = 0
    total_loss = 0

    model.eval()

    for itr, batch in enumerate(tqdm(data_loader, desc='Test')):
        inputs, labels = batch
        labels = labels.to(device)

        input_values = inputs.to(device)
        outputs = model(input_values)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.shape[0]
        total_loss += loss.item()

    accuracy = 100 * total_correct / total_samples
    logger.info(f'[test] loss: {total_loss / total_samples:.3f}, acc: {accuracy:.2f}%')
    return loss, accuracy

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # hyper parameters
    batch_size = 256
    learning_rate = 1e-3
    total_epochs = 50

    # load dataset
    logger.info(f'Loading dataset...')
    npz_dataset_train = np.load('/raid/yixu/Projects/Speech/proj/dataset_train.npz')
    npz_dataset_val = np.load('/raid/yixu/Projects/Speech/proj/dataset_val.npz')
    npz_dataset_test = np.load('/raid/yixu/Projects/Speech/proj/dataset_test.npz')
    num_labels = 61

    # create the dataloader
    train_dataset = TimitDataset(npz_dataset_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimitDataset(npz_dataset_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TimitDataset(npz_dataset_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # create the model
    layer_list = [512, 256]
    model = FrameLevelPhonemeClassifier(num_labels=num_labels, layer_list=layer_list).to(device)
    best_model_path = None

    # # prepare the process (wav2vec2)
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    save_model_interval = 3
    cur_version = datetime.now().strftime('%y%m%d-%H%M%S')
    model_root = f'results/{cur_version}'
    os.makedirs(model_root, exist_ok=True)
    config = {
        'batch_size': batch_size,
        'lr': learning_rate,
        'total_epochs': total_epochs,
        'layer_list': layer_list,
    }
    config_file = os.path.join(model_root, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)
    logger.info(f'Start a new training with version "{cur_version}"')

    # patience (early stopping patience)
    patience = 5
    patience_cnt = 0
    best_acc = 0

    # register a wandb logger
    wandb.login()
    run = wandb.init(
        project="speech-group-project",
        config=config,
    )

    # training loop
    for epoch in range(total_epochs):

        # training
        train(model, train_loader, criterion, optimizer, epoch, total_epochs, device)

        # validation
        val_loss, val_acc = val(model, val_loader, criterion, epoch, total_epochs, device)

        wandb.log({'val-epoch': epoch, 'val-loss': val_loss, 'val-accuracy': val_acc})

        # update learning rate
        scheduler.step()

        # Save the best model
        if epoch != 0 and epoch % save_model_interval == 0:
            model_path = os.path.join(model_root, f'epoch{epoch:03d}-loss{val_loss:.3f}-acc{val_acc:.2f}.pt')
            if best_model_path is None or best_acc > val_acc:
                torch.save({
                            'state_dict': model.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            'acc': val_acc
                            }, model_path)
                if best_model_path is not None:
                    os.remove(best_model_path)
                best_model_path = model_path
            logger.info(f'Save the best model in "{model_path}"')
        
        # check early stopping
        if val_acc > best_acc:
            best_acc = val_acc
        else:
            patience_cnt += 1
        
        if patience_cnt == patience:
            logger.warning(f'Early stop because of exceeding patience {patience}.')
            break

    # test the model
    test_loss, test_acc = test(model, test_loader, criterion, device)
    final_model_path = Path(best_model_path).parent / f'{Path(best_model_path).stem}_testloss{test_loss:.3f}_testacc{test_acc:.2f}.pt'
    shutil.move(best_model_path, final_model_path)

if __name__ == '__main__':
    main()

'''
Usage:

# data preprocessing
cd dataset && python timit.py

# training command:
python train.py

# Monitor the training process by wandb (need to login by github account)
https://wandb.ai/544200536/speech-group-project
'''