import os
import numpy as np
import cv2
import torch
from torchvision import transforms

class MiniBatchLoader(object):
 
    def __init__(self, train_path, test_path, image_dir_path, img_length, img_width):
 
        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
 
        self.img_length = img_length
        self.img_width = img_width
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)
 
    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
 
    
    # test ok
    
    
    
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        path_list = []
        if mini_batch_size>1:    
            masks = np.zeros((mini_batch_size,1,256,384))
            raw_x = np.zeros((mini_batch_size,3,256,384))
            for i, index in enumerate(indices):
                path = path_infos[index]
                mask_path = path.replace('.jpg','_gt.png')
                #print(mask_path)
                img = cv2.imread(path).astype(np.float32)[..., ::-1]
                #img = cv2.imread(path).astype(np.float32)
                mask = cv2.imread(mask_path,0)
                #print(np.unique(mask))
                mask = mask.astype(np.float32)
                #print(np.unique(mask))
                mask =  mask / 255.
                #print(np.unique(mask))
                raw_x[i,:,:,:]  = img.transpose(2,0,1)
                #cv2.imwrite("./ori/"+path.split('/')[-1],img)
                #cv2.imwrite("./trans/"+path.split('/')[-1], raw_x[i,:,:,:].transpose(1,2,0))
                #img = img.transpose(2,0,1)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                #raw_x[i,:,:,:] = img
                masks[i,0,:,:] = mask
                #print(np.unique(masks[i,0,:,:]))
                path_list.append(path)
        else: 
            in_channels = 3
            
            path = path_infos[indices[0]]
            if '.jpg' in path:
                mask_path=path.replace('.jpg','_gt.png')
            if '.png' in path:
                mask_path=path.replace('.png','_gt.png')
            if '.tif' in path:
                mask_path = path.replace('.tif','_gt.png')
            #mask_path = path.replace('.jpg','_gt.png')
            img = cv2.imread(path).astype(np.float32)[..., ::-1]
            #masks = np.zeros((mini_batch_size, 1, img.shape[0], img.shape[1])).astype(np.float32)
        
            #raw_x = np.zeros((mini_batch_size, in_channels, img.shape[0], img.shape[1])).astype(np.float32)
            # if not img.shape[0] %2 ==0:
                # new_img = np.zeros((img.shape[0]+15,img.shape[1],3)).astype(np.float32)
                # new_img[0:img.shape[0],:,:] = img
                # img = new_img.copy()
            # if not img.shape[1] %2 ==0:
                # new_img = np.zeros((img.shape[0],img.shape[1]+15,3)).astype(np.float32)
                # new_img[:,0:img.shape[1],:] = img
                # img = new_img.copy()
            mask = cv2.imread(mask_path,0).astype(np.float32) / 255.
            print(mask.size) 
            if img is None:
                raise RuntimeError("invalid image")
            raw_x = img.transpose(2,0,1)
            raw_x = raw_x[np.newaxis,:,:,:]
            #raw_x[0,:,:,:] = img
            #img = img.transpose(2,0,1)
            masks = mask[np.newaxis,np.newaxis,:,:]
            path_list.append(path)
 
        return  masks, path_list, raw_x
