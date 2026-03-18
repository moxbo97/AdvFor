
import torch
import numpy as np
import cv2
from tqdm import tqdm
import State as State
from pixelwise_a3c import *
from FCN import *
from reward import *
from mini_batch_loader import MiniBatchLoader
from osn import *
import matplotlib.pyplot as plt
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
MOVE_RANGE = 27
EPISODE_LEN = 7
MAX_EPISODE = 100000
GAMMA = 0.90
N_ACTIONS = 3
BATCH_SIZE = 4
DIS_LR = 3e-4
LR = 1e-3
img_length = 256
img_width = 384
img_channel = 3
sigma = 25
TRAINING_DATA_PATH = "casia_train_512.txt"
TESTING_DATA_PATH = "imd.txt"
SAVING_EPISODE = 1000
IMAGE_DIR_PATH = ""
SAVE_PATH = "./ANTI_OSN_MIN/pre_msk/"
from torchvision import transforms

def img_transform(batch_img):
    norm_img = torch.from_numpy(np.zeros(batch_img.shape,dtype=np.float32))
    transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    for i in range(batch_img.shape[0]):
        img = batch_img[i,:,:,:]
        img = transform((img.astype(np.float32)/255.).transpose(1,2,0))
        norm_img[i,:,:,:] = img
        
    return norm_img
    
def main():
    model = EfficientUnet(N_ACTIONS).to(device)
    #model.load_state_dict(torch.load(MODEL_PATH))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    i_index = 0
    
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        img_length,
        img_width)

    current_state = State.State((BATCH_SIZE, img_channel, img_length, img_width), MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, BATCH_SIZE, EPISODE_LEN, GAMMA)
    noise_intensity = 0.15
    wa = 1.
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    osn_model = load_osn()
    pre_f1 = 0
    pre_iou = 0
    pre_auc = 0
    cur_f1 = 0
    cur_iou=0
    cur_auc = 0
    sum_f1 = 0
    sum_iou = 0
    sum_auc = 0
    print("\n")
    for n_epi in tqdm(range(0, MAX_EPISODE), ncols=70, initial=0):
        print("\n")
        r = indices[i_index: i_index + BATCH_SIZE]
        mask, path, raw_x = mini_batch_loader.load_training_data(r)
        norm_img = img_transform(raw_x)
        action = np.zeros((BATCH_SIZE, img_length, img_width))
        current_state.reset(raw_x) 
        reward = np.zeros(raw_x.shape, np.float32) 
        sum_reward = 0
        init_mask = np.round(osn_model(norm_img.cuda()).cpu().detach().numpy())
        init_f1, init_iou, init_auc = get_f1_and_iou_and_auc(mask, init_mask)
        print("Init:  F1: "+str(init_f1)+" IOU: "+str(init_iou)+" AUC: "+str(init_auc))
        current_mask = init_mask.copy()
        best_f1 = init_f1
        best_iou = init_iou
        best_auc = init_auc

        for t in range(EPISODE_LEN):
            previous_img = current_state.image.copy() 
            previous_mask = current_mask.copy()
            action = agent.act_and_train(current_state.state, reward) 
            move_map = current_state.step(action,noise_intensity*0.8**t)
            current_img = current_state.image.copy()
            current_norm_img = img_transform(current_img)
            current_mask = np.round(osn_model(current_norm_img.cuda()).cpu().detach().numpy())
            cur_f1, cur_iou, cur_auc = get_f1_and_iou_and_auc(mask, current_mask)
            if best_f1>cur_f1:
                best_f1=cur_f1
                best_iou=cur_iou
                best_auc=cur_auc
            print("Iter: "+str(t)+" F1: "+str(cur_f1)+" IOU: "+str(cur_iou)+" AUC: "+str(cur_auc))
            reward = get_reward(init_mask, current_mask, current_img, raw_x, previous_img, wa) 
            print("Reward: "+str(np.mean(reward)))
            sum_reward += np.mean(reward) * np.power(GAMMA, t)
        sum_f1+=best_f1
        sum_iou+=best_iou
        sum_auc+=best_auc
        print("AVERAGE   F1: "+str(sum_f1/np.float(n_epi%200+1))+" IOU: "+str(sum_iou/np.float(n_epi%200+1))+" AUC: "+str(sum_auc/np.float(n_epi%200+1)))
        if (n_epi+1)%200 == 0:
            if sum_f1/np.float(n_epi%200+1)<0.3 and noise_intensity>0.005:
                print(sum_f1/np.float(n_epi%200+1))
                noise_intensity *=0.99
            print(noise_intensity)
            if sum_f1/np.float(n_epi%200+1)<0.3 and wa<20:
                wa*=1.01
            sum_f1 = 0
            sum_iou = 0
            sum_auc = 0
        agent.stop_episode_and_train(current_state.state, reward, True)
        if i_index + BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += BATCH_SIZE

        if i_index + 2 * BATCH_SIZE >= train_data_size:
            i_index = train_data_size - BATCH_SIZE

        for i in range(len(path)):
            img_name = path[i].split('/')[-1]
            
        print("train total reward {a}".format(a=sum_reward ))
        if n_epi % 400 == 0:
            torch.save(model.state_dict(), SAVE_PATH+str(n_epi)+'.pth')
def paint_amap(acmap):
    image = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    # print(image)
    plt.imshow(image, vmin=1, vmax=9)
    plt.colorbar()
    plt.pause(1)
    # plt.show()
    plt.close()
if __name__ == '__main__':
    main()
