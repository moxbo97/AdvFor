import os
import cv2
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from models.scse import SCSEUnet


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.name = 'detector'
        self.det_net = SCSEUnet(backbone_arch='senet154', num_channels=3)

    def forward(self, Ii):
        Mo = self.det_net(Ii)
        return Mo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.save_dir = 'weights/'
        self.networks = Detector()
        self.gen = nn.DataParallel(self.networks).cuda()
        self.bce_loss = nn.BCELoss()

    def forward(self, Ii):
        return self.gen(Ii)

    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + '%s_weights.pth' % self.networks.name))


def load_osn():
    model = Model().cuda()
    model.load()
    model.eval()
    return model

