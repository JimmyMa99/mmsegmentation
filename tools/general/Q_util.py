import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries
import cv2
import sys
init_turn_grid =None
import math
# import pytorch_colors as colors
def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    h_per_patch = h // num_pieces_per_line
    w_per_patch = w // num_pieces_per_line
    features = features[:,:,:h_per_patch*num_pieces_per_line,:w_per_patch*num_pieces_per_line]
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)
def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]
    if(False):
        loss_map_sem = reconstr_feat[:, :-2, :, :] - labxy_feat[:, :-2, :, :]
        loss_sem =  torch.norm(loss_map_sem, p=2, dim=1).sum() / (b * S)
    else:
        # self def cross entropy  -- the official one combined softmax
        logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
        loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b    
        
  
    
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sem_sum =   0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos
    loss_sum =   loss_pos_sum +loss_sem_sum


    return loss_sum, loss_sem_sum,  loss_pos_sum

def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):

        ext_w_list = []
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])
            index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features


def init_spixel_grid(args,  b_train=True):
    if b_train:
        img_height, img_width,batch_size= args.image_size, args.image_size,args.batch_size
    else:
        img_height, img_width,batch_size = args.image_size, args.image_size,args.batch_size

    # get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height/args.downsize))
    n_spixl_w = int(np.floor(img_width/args.downsize))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor =  np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height,axis=1), spixel_width, axis=2)

    torch_spix_idx_tensor = torch.from_numpy(
                np.tile(spix_idx_tensor, (batch_size, 1, 1, 1))).type(torch.float).cuda()


    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # pixel coord
    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
  
    all_XY_feat = (torch.from_numpy(
        np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).cuda())

    return  torch_spix_idx_tensor, all_XY_feat

#===================== pooling and upsampling feature ==========================================

def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor


def poolfeat(input, prob, sp_h=16, sp_w=16):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape 
    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda(input.device)], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) # b * (n+1) * h* w
    send_to_top_left =  F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top )

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)


    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat

def upfeat(input, prob, up_h=16, up_w=16):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)


    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def build_LABXY_feat(label_in, XY_feat):

    img_lab = label_in.clone().type(torch.float)

    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img =  F.interpolate(img_lab, size=(curr_img_height,curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat],dim=1)

    return LABXY_feat

def label2one_hot_torch(labels, C=14):
    # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
    '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
    b,_, h, w = labels.shape
    one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda(labels.device)
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1) #require long type

    return target.type(torch.float32)


def refine_with_affmat(input,affmat):
    b, c, h, w = input.shape

    h_shift = 2
    w_shift = 2

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)
    cat_mat=[]
    for i in range(5):
         for j in range(5):
             cat_mat.append(feat_pd[:,:,i:i+h,j:j+w])
    cat_mat=torch.stack(cat_mat,dim=1)
    return torch.sum(cat_mat*affmat.reshape(b,25,1,h,w),dim=1)

    

def get_turn(area=5):
    def get_in(index):
        centerlist=[(index-10)%25,(index-5)%25,index,(index+5)%25,(index+10)%25]
        retlist=[]
        for x in centerlist:
            lll = int(x/5)*5
            retlist+=[(x-2+5)%5+lll,(x-1+5)%5+lll,x,(x+1)%5+lll,(x+2)%5+lll]
        return torch.tensor(retlist)
    turn_grid=[]
    for i in range(25):
        turn_grid.append(get_in(i))

    return torch.stack(turn_grid)

def calc_affmat(psam,down_size=16):

        init_turn_grid=get_turn().reshape(1,25,5,5).cuda(psam.device)
        b,c,h,w=psam.shape
        ini_grid=torch.arange(0,25,1).reshape(1,1,5,5).cuda(psam.device)
        ini_grid=label2one_hot_torch(ini_grid,25)
        ini_grid=ini_grid.repeat(b,1,int(h/(down_size*5))+1,int(w/(down_size*5))+1)[:,:,:int(h/down_size),:int(w/down_size)].detach()

        turn_grid=init_turn_grid.repeat(b,1,int(h/(down_size*5))+1,int(w/(down_size*5))+1)[:,:,:int(h/down_size),:int(w/down_size)]

        up_ini_grid= upfeat(ini_grid,psam,8,8)
        aff_grid = poolfeat(up_ini_grid,psam,8,8)  
        aff_mat = torch.gather(aff_grid,1,turn_grid)
        
        return aff_mat
def refine_with_q(input,prob,iter=20,aff_mat= None,down_size=16,with_aff=False):
    if(iter>0 or with_aff):
        if(aff_mat==None):   
            aff_mat = calc_affmat(prob)
        if(input==None): 
            return None,aff_mat
        if(iter>0):
            if(prob.shape[2]==input.shape[2]):
                input= poolfeat(input,prob,down_size,down_size)
                for i in range(iter-1):
                    input= refine_with_affmat(input,aff_mat)
                input = upfeat(input,prob)
            else:
                for i in range(iter):
                    input= refine_with_affmat(input,aff_mat)
        if(with_aff):
            return input,aff_mat

    return input

def pool_feat_2(probs,feats):
    b,cp,h,w=probs.shape
    b,cf,h,w=feats.shape
    probs=probs.view(b,cp,h*w).transpose(1,2)
    probs_sum=torch.sum(probs,dim=1,keepdim=True)
    feats=feats.view(b,cf,h*w)
    ret=torch.bmm(feats,probs)/(probs_sum+1e-5)
    return ret
def up_feat_2(probs,feats_min):
    b,cp,h,w=probs.shape
    b,cf,cp=feats_min.shape
    probs=probs.view(b,cp,h*w)
    ret=torch.bmm(feats_min,probs).view(b,cf,h,w)
    return ret
