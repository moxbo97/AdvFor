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
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

MOVE_RANGE = 3
EPISODE_LEN = 3
TESTING_MAX_EPISODE_LEN = 7
GAMMA = 0.8
N_ACTIONS = 3
BATCH_SIZE = 1
DIS_LR = 3e-4
LR = 1e-3
img_length = 384
img_width = 256
img_channel = 3
sigma = 25
TRAINING_DATA_PATH = "./casia_train_512.txt"
TESTING_DATA_PATH = "./ipm.txt"
SAVING_EPISODE = 1000
IMAGE_DIR_PATH = ""

SAVE_PATH = "/data2/moxb/BIFF/ANTI_OSN/curbest/5000.pth"
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
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    #iid = load_iid()
    osn_model = load_osn()
    init_f1_list = []
    init_iou_list = []
    init_auc_list = []
    adv_f1_list = []
    adv_iou_list = []
    adv_auc_list = []
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        384,
        256)

    agent = PixelWiseA3C_InnerState(model, optimizer, BATCH_SIZE, EPISODE_LEN, GAMMA)

    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    indices = np.random.permutation(test_data_size)
    
    best_f1 = 1
    cur_auc = 0

    
    query = []
    psnr_list = []
    ssim_list = []
    for i_index in tqdm(range(0, test_data_size), ncols=70, initial=0):

        r = indices[i_index: i_index + BATCH_SIZE]

        mask, path, raw_x = mini_batch_loader.load_testing_data(r)
        if len(np.unique(mask))!=2:
            continue
        img_channel = raw_x.shape[1]
        img_length = raw_x.shape[2]
        img_width = raw_x.shape[3]
        best_img = np.zeros((BATCH_SIZE, img_channel, img_length, img_width), dtype=np.float32)
        best_msk = np.zeros((BATCH_SIZE, 1, img_length, img_width), dtype=np.float32)
        current_state = State.State((BATCH_SIZE, img_channel, img_length, img_width), MOVE_RANGE)
        #print(np.max(raw_x))
        img_name = path[0].split('/')[-1]
        print(img_name)
        norm_img = img_transform(raw_x.copy())
        #print(norm_img.shape)
        current_state.reset(raw_x.copy())
        # print(np.max(raw_x))
        sum_reward = 0
        init_mask = osn_model(norm_img.cuda()).cpu().detach().numpy()
        current_mask = init_mask.copy()
        init_f1, init_iou, init_auc = get_f1_and_iou_and_auc(mask, init_mask)
        best_f1 = init_f1
        best_iou = init_iou
        best_auc = init_auc
        init_f1_list.append(init_f1)
        init_iou_list.append(init_iou)
        init_auc_list.append(init_auc)
        best_f1 = init_f1 
        print("Init:  F1: "+str(init_f1)+" IOU: "+str(init_iou)+" AUC: "+str(init_auc))
        best_img = current_state.image.copy()
        best_msk = init_mask.copy()
        for t in range(TESTING_MAX_EPISODE_LEN):
            action = agent.act(current_state.state)

            move_map = current_state.step(action, 0.045*0.8**t)
            #print(np.max(raw_x))
            
            #reward = get_iid_reward(previous_gradient, move_map)
            current_img = current_state.image.copy()
            current_norm_img = img_transform(current_img)
            current_mask = osn_model(current_norm_img.cuda()).cpu().detach().numpy()

            current_f1, current_iou, current_auc = get_f1_and_iou_and_auc(mask, current_mask)
            print("Iter: "+str(t+1)+" F1: "+str(current_f1)+" IOU: "+str(current_iou)+" AUC: "+str(current_auc)) 

            if best_f1>current_f1:
                best_f1 = current_f1
                best_iou = current_iou
                best_auc = current_auc
                best_img, best_msk = current_state.image.copy(), current_mask.copy()
            if not os.path.exists("./AdvFor/OSN/COL/"+str(t)+'/img/'):
                os.makedirs("./AdvFor/OSN/COL/"+str(t)+'/img/')
            cv2.imwrite("./AdvFor/OSN/COL/"+str(t)+'/img/'+img_name, current_state.image[0,:,:,:].transpose(1,2,0)[...,::-1].astype(np.uint8))
            if not os.path.exists("./AdvFor/OSN/COL/"+str(t)+'/msk/'):
                os.makedirs("./AdvFor/OSN/COL/"+str(t)+'/msk/')
            cv2.imwrite("./AdvFor/OSN/COL/"+str(t)+'/msk/'+img_name, np.round(current_mask[0,0,:,:])*255.)

        adv_f1_list.append(best_f1)
        adv_iou_list.append(best_iou)
        adv_auc_list.append(best_auc)
        
        if not os.path.exists("./AdvFor/original/COL/img/"):
            os.makedirs("./AdvFor/original/COL/img/")
        if not os.path.exists("./AdvFor/original/COL/msk/"):
            os.makedirs("./AdvFor/original/COL/msk/")
        if not os.path.exists("./AdvFor/original/COL/gt/"):
            os.makedirs("./AdvFor/original/COL/gt/")
        cv2.imwrite("./AdvFor/original/COL/img/"+img_name,raw_x[0,:,:,:].transpose(1,2,0)[...,::-1].astype(np.uint8))
        cv2.imwrite("./AdvFor/original/COL/gt/"+img_name, mask[0,0,:,:]*255.)
        cv2.imwrite("./AdvFor/original/COL/msk/"+img_name, np.round(init_mask[0,0,:,:])*255.)

        if not os.path.exists("./AdvFor/OSN/COL/img/"):
            os.makedirs("./AdvFor/OSN/COL/img/")
        if not os.path.exists("./AdvFor/OSN/COL/msk/"):
            os.makedirs("./AdvFor/OSN/COL/msk/")
        
        cv2.imwrite("./AdvFor/OSN/COL/img/"+img_name, best_img[0,:,:,:].transpose(1,2,0)[...,::-1].astype(np.uint8))
        #print((best_img))
        cv2.imwrite("./AdvFor/OSN/COL/msk/"+img_name, np.round(best_msk[0,0,:,:])*255.)
        ori_img = raw_x[0,:,:,:].transpose(1,2,0)[...,::-1].copy().astype(np.uint8)
        #print(np.max(ori_img))
        
        adv_img = best_img[0,:,:,:].transpose(1,2,0)[...,::-1].astype(np.uint8)
        
        #print(np.max(adv_img))
        MSE = mean_squared_error(ori_img, adv_img)
        PSNR = peak_signal_noise_ratio(ori_img, adv_img)
        SSIM = structural_similarity(ori_img, adv_img, multichannel=True)
        if SSIM<1.0:
            ssim_list.append(SSIM)
            psnr_list.append(PSNR)
        print("MSE:"+str(MSE)+" PSNR:"+str(PSNR)+" SSIM:"+str(SSIM))
        #print("train total reward {a}".format(a=sum_reward ))
        print("Original F1:"+str(np.mean(init_f1_list))+" IOU: "+str(np.mean(init_iou_list))+" AUC: "+str(np.mean(init_auc_list)))
        print("AdvFor F1:"+str(np.mean(adv_f1_list))+" IOU: "+str(np.mean(adv_iou_list))+" AUC: "+str(np.mean(adv_auc_list)))
        print("AVERAGE PSNR: "+str(np.mean(psnr_list))+" SSIM: "+str(np.mean(ssim_list)))

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
