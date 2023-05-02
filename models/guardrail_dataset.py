import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import cv2

# transformer
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.9),
    transforms.RandomAffine(degrees=0, translate=(0.005, 0.0025)),
])
transform_toTensor = transforms.Compose([
    transforms.ToTensor()
])

# def image2label(img, cm2lbl):
#     data = np.array(img, dtype='int32')
#     idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
#   # print(np.unique(idx))
#   # print(cm2lbl[idx])
#     result = np.array(cm2lbl[idx], dtype='int64')  
#     return result[:,:,None]

class IS_Dataset(Dataset):
    CLASSES = ['background', 'girder', 'net', 'lanyard', 'guardrail']
    def __init__(self, data_list, label_list, classes, cm2lbl, transform=None, transform_toTensor=None, resize_size=(512, 512)):

        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform
        self.transform_toTensor = transform_toTensor
        self.cm2lbl = cm2lbl
        self.resize_size = resize_size
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        print('Read ' + str(len(self.data_list)) + ' images')
        
    def image2label(self, img, cm2lbl):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
      # print(np.unique(idx))
      # print(cm2lbl[idx])
        result = np.array(cm2lbl[idx], dtype='int64')  
        return result[:,:,None]

    def __getitem__(self, index):
        # read data

        image = cv2.imread(self.data_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize_size, interpolation=cv2.INTER_AREA)

        mask = cv2.imread(self.label_list[index])
        mask = cv2.resize(mask, self.resize_size, interpolation=cv2.INTER_NEAREST)

        # apply data augmentations
        if self.transform:
          # ndarray to PIL image
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
          # apply the transforms
            image = self.transform(image)
            mask = self.transform(mask)

        #把RGB改成0,1,...,class_num : (w, w, 1)
        mask = self.image2label(mask, self.cm2lbl)
        mask = mask.squeeze()
        # print(np.unique(mask))
        # print(mask.shape)

        # 把每個不同目標分成0,1 : (w, w, class_num)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.transform_toTensor:
            image = self.transform_toTensor(image)
            mask = self.transform_toTensor(mask)

        return image, mask
    
    def __len__(self):
        return len(self.data_list)

    
# class IS_Dataset_test(Dataset):
#     CLASSES = ['background', 'girder', 'net', 'lanyard', 'guardrail']
#     def __init__(self, data_list, label_list, classes, cm2lbl, transform=None, transform_toTensor=None, resize_size=None):

#         self.data_list = data_list
#         self.label_list = label_list
#         self.transform = transform
#         self.transform_toTensor = transform_toTensor
#         self.cm2lbl = cm2lbl
#         self.resize_size = resize_size
#         # convert str names to class values on masks
#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
#         print('Read ' + str(len(self.data_list)) + ' images')

#     def __getitem__(self, index):
#         # read data
#         image = cv2.imread(self.data_list[index])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     image = cv2.resize(image, self.resize_size, interpolation=cv2.INTER_AREA)

#         mask = cv2.imread(self.label_list[index])
#     #     mask = cv2.resize(mask, self.resize_size, interpolation=cv2.INTER_NEAREST)
#         if self.resize_size:
#             image = cv2.resize(image, self.resize_size, interpolation=cv2.INTER_AREA)
#             mask = cv2.resize(mask, self.resize_size, interpolation=cv2.INTER_NEAREST)
#         if image.shape[0] > 1400 or image.shape[1] > 1400 :
#             ratio = ratioFunction(image.shape[0], image.shape[1])
#             new_size = (960, int(960*ratio))
#             image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
#             mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
#         if image.shape[0]%16 !=0 or image.shape[1]%16 !=0 :
#             new_size = (960, 720)
#             image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
#             mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
#         print(image.shape)
#         # apply data augmentations
#         if self.transform:
#           # ndarray to PIL image
#             image = Image.fromarray(image)
#             mask = Image.fromarray(mask)
#             # apply the transforms
#             image = self.transform(image)
#             mask = self.transform(mask)

#         #把RGB改成0,1,...,class_num : (w, w, 1)
#         mask = image2label(mask, self.cm2lbl)
#         mask = mask.squeeze()
#         # print(np.unique(mask))
#         # print(mask.shape)

#         # 把每個不同目標分成0,1 : (w, w, class_num)
#         masks = [(mask == v) for v in self.class_values]
#         mask = np.stack(masks, axis=-1).astype('float')

#         if self.transform_toTensor:
#             image = self.transform_toTensor(image)
#             mask = self.transform_toTensor(mask)

#         return image, mask
    
#     def __len__(self):
#         return len(self.data_list)
