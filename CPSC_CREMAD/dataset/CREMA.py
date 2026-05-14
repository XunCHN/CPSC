import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import torch
from torchvision import transforms
from typing import Tuple
import csv
import torchvision
from torch import Tensor
import librosa

class CramedDataset(Dataset):

    def __init__(self, config, mode='train'):
        self.config = config
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.use_pre_frame = 3
        self.data_root = config["dataset"]["data_root"]
        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.train_csv = 'train.csv'
        self.test_csv = 'test.csv'

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            # n=0
            for item in csv_reader:
                audio_path = os.path.join('cremad_datasets', 'AudioWAV', item[0] + '.wav')
                visual_path = os.path.join('cremad_datasets', 'Image', item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                    # n+=1
                    # if n==100:
                    #     break
                else:
                    continue

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 20)[:22050 * 20]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = torch.tensor(spectrogram)
        #print(spectrogram.shape)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        
        file_num = len(image_samples)
        #print(file_num)
        
        if file_num < self.use_pre_frame:
            select_index = random.choices(image_samples, k=self.use_pre_frame)
        else:
            select_index = random.sample(image_samples, self.use_pre_frame)
        #print(len(select_index))       
        
        select_index.sort()
        images = torch.zeros((self.use_pre_frame, 3, 224, 224))
        
        
        for i in range(self.use_pre_frame):
            img = Image.open(os.path.join(self.image[idx], select_index[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))
        one_hot = np.eye(self.config["setting"]["num_class"])
        one_hot_label = one_hot[self.label[idx]]
        label = torch.FloatTensor(one_hot_label)

        return spectrogram, images, label
