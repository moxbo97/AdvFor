from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import roc_auc_score
import numpy as np
from torchvision import transforms
import torch
transform = transforms.Compose([
             np.float32,
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

def gen_adv_samples(iid, image, mask):
    img_np = image.copy()
    msk_np = mask.copy()/255.
    image = torch.tensor(image/255.,dtype=torch.float)
    #image = image.permute(0,3,1,2)
    image = torch.autograd.Variable(image, requires_grad=True)
    image.retain_grad()
    mask = torch.tensor(mask/255.,dtype=torch.float)
    mo, lo = iid.process(image.cuda(),mask.cuda())
    f1, iou, auc = get_f1_and_iou_and_auc(msk_np, mo.cpu().detach().numpy())
    print("ORI F1: "+str(f1)+ " IOU: "+str(iou)+" AUC: "+str(auc))
    lo.backward(retain_graph=False)
    gradients = image.grad
    noise = np.sign(gradients.cpu().detach().numpy()).astype(np.float)*0.008
    img_np = img_np/255.+noise
    img_np = img_np*255.
    img_np = np.clip(img_np, a_min=0., a_max=255)
    
    return img_np
    

def get_reward(pre_msk, cur_msk, img, ori_img, pre_img, wa):
    #target = np.ones(cur_msk.shape)
    #pre_msk = pre_msk*gt
    #cur_msk = cur_msk*gt
    #gt = np.zeros(pre_msk.shape)
    #print(np.max(cur_msk))
    #print(np.min(cur_msk))
    #weight = np.ones(gt.shape)*100.
    #weight[weight==0] = 1
    #return np.tanh(np.power((pre_msk - gt), 2) - np.power((cur_msk - gt), 2))
    #reward = np.power((cur_msk),2)
    distortion = np.tanh(np.power((ori_img-img)/255.,2))
    pre_distortion = np.tanh(np.power((ori_img - pre_img)/255.,2))
    wa = 1
    adv_reward = (pre_msk-cur_msk).astype(np.float)*10
    cost_reward = pre_distortion-distortion
    print('adv_reward:'+str(np.mean(adv_reward))+'   cost_reward:'+str(np.mean(cost_reward)))
    return (pre_msk-cur_msk).astype(np.float)*10-(pre_distortion-distortion)*wa
    #return ((cur_msk<0.5).astype(np.float))*100-distortion*10

def get_loss_reward(pre_msk, cur_msk, gt, init_msk):
    pre_dist = np.power((pre_msk - gt),2)
    cur_dist = np.power((cur_msk - gt),2)
    reward = cur_dist - pre_dist
    #print(np.mean(reward))
    pre_False_Pred = np.round(pre_msk) != gt
    cur_False_Pred = np.round(cur_msk) != gt
    reward_weight_positive = np.logical_and(cur_False_Pred, (pre_False_Pred == False)).astype(np.float64)*100
    reward_weight_negative = np.logical_and(pre_False_Pred, (cur_False_Pred == False)).astype(np.float64)*100
    reward_weight_positive[reward_weight_positive==0] = 1
    reward_weight_negative[reward_weight_negative==0] = 1
    for i in range(reward.shape[0]):
        tmp = reward[i,0,:,:]
        r_max = tmp.max()
        r_min = tmp.min()
        r_mean = tmp.mean()
        norm_tmp = (tmp-r_mean)/(r_max-r_min)
        reward[i,0,:,:] = norm_tmp
    reward = reward * reward_weight_positive * reward_weight_negative
    pixel_wise_reward = np.ones([reward.shape[0],3,reward.shape[2],reward.shape[3]],dtype=np.float32)
    pixel_wise_reward[:,0,:,:] = reward[:,0,:,:]
    pixel_wise_reward[:,1,:,:] = reward[:,0,:,:]
    pixel_wise_reward[:,2,:,:] = reward[:,0,:,:]

    return pixel_wise_reward

def get_visual_reward(cur_img, pre_img, ori_img):
    cur_dist = np.power((cur_img - ori_img),2)
    pre_dist = np.power((pre_img - ori_img),2)
    reward = pre_dist - cur_dist
    return reward

def get_f1_and_iou_and_auc(gts,predicts):
    f1, iou, auc = [],[],[]
    H,W,C = gts.shape[2],gts.shape[3],gts.shape[1]
    for i in range(gts.shape[0]):
        groundtruth = gts[i,:,:,:]
        predict_mask = predicts[i,:,:,:]
        auc.append(
                        roc_auc_score(
                            (groundtruth.reshape(H * W * C)).astype(np.int32),
                            predict_mask.reshape(H * W * C) ,
                        )
                    )
        
        predict_mask = np.round(predict_mask)
        predict_mask = predict_mask>0
        groundtruth = groundtruth>0
        seg_inv = np.logical_not(predict_mask)
        gt_inv = np.logical_not(groundtruth)
        true_pos = float(np.logical_and(predict_mask, groundtruth).sum())  # float for division
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(predict_mask, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, groundtruth).sum()
        f1.append(2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6))
        cross = np.logical_and(predict_mask, groundtruth)
        union = np.logical_or(predict_mask, groundtruth)
        tmp_iou = np.sum(cross) / (np.sum(union) + 1e-6)
        if np.sum(cross) + np.sum(union) == 0:
            iou.append(1)
        else:
            iou.append(tmp_iou)
        
    return np.mean(f1),np.mean(iou), np.mean(auc)
            
