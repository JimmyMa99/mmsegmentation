import core.models as fcnmodel
from tools.ai.torch_utils import *
from tools.general.Q_util import *
from torch import nn

from torchvision import transforms
from tools.ai.augment_utils import Normalize_For_inference, Transpose_For_Segmentation

import torch.nn.functional as F

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

class ASAMrefiner():
    def __init__(self,model_path,transform=None) -> None:
        self.Q_model = fcnmodel.SpixelNet1l_bn().cuda()
        self.Q_model = nn.DataParallel(self.Q_model)
        load_model(self.Q_model, model_path, parallel=1)
        self.Q_model.eval()
        if transform is None:
            transform = transforms.Compose([
                Normalize_For_inference(imagenet_mean, imagenet_std),
                Transpose_For_Segmentation()
            ])
        self.transform = transform
        self.down_size=16
    
    def calc_affmatmask(self,psam):

        init_turn_grid=get_turn().reshape(1,25,5,5).cuda(psam.device)

    def get_affmats_cammask(self,img): #size of img is equal to cam
        Qs = self.Q_model(img)
        affmats = calc_affmat(Qs)
        return affmats


    def get_affmats(self,img): #size of img is not equal to cam
        img = F.interpolate(img, size=(img.shape[2]//16*16,img.shape[3]//16*16), mode='bilinear', align_corners=False)
        Qs = self.Q_model(img)
        affmats = calc_affmat(Qs)
        return affmats

    def refine(self,cam,affmats,renum):
        b,c,h,w=cam.shape
        aff_b,aff_c,aff_h,aff_w=affmats.shape
        if h!=aff_h or w!=aff_w:
            cam = F.interpolate(cam, size=(aff_h,aff_w), mode='bilinear', align_corners=False)
        for i in range(renum):
            cam = refine_with_affmat(cam, affmats)
        return cam
    
    def to_tensor(self,data):
        return torch.from_numpy(data['image']).float().cuda().unsqueeze(0)
    
    def data_preprocess(self,img):
        inputs=self.transform(img)
        inputs = self.to_tensor(inputs)
        return inputs


    def __call__ (self,img,cam,renum):
        inputs=self.data_preprocess(img)
        b,c,H,W=inputs.shape
        l,h,w=cam.shape
        tensor_cam = torch.from_numpy(cam).float().cuda().unsqueeze(0)
        if h==H and w==W:
            #affmats = self.get_affmats_cammask(inputs)
            cam = F.interpolate(tensor_cam, size=(h//16,w//16), mode='bilinear', align_corners=False)
        affmats = self.get_affmats(inputs)
        cam = self.refine(cam,affmats,renum)
        cam = F.interpolate(cam, size=(H,W), mode='bilinear', align_corners=False)
        cam = cam.detach().squeeze(0).cpu().numpy()

        return cam #(2, 336, 500)

    def run_infer(self,img,cam,renum):
        inputs=img
        b,c,H,W=inputs.shape
        b,c,h,w=cam.shape
        tensor_cam = cam
        if h!=H//16 and w!=W//16:
            #affmats = self.get_affmats_cammask(inputs)
            cam = F.interpolate(tensor_cam, size=(H//16,W//16), mode='bilinear', align_corners=False)
        affmats = self.get_affmats(inputs)
        cam = self.refine(cam,affmats,renum)
        cam = F.interpolate(cam, size=(h,w), mode='bilinear', align_corners=False)


        return cam #(2, 336, 500)

