import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_bcss.pth', type=str)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--dataroot", default="datasets/BCSS-WSSS/", type=str)
    parser.add_argument("--dataset", default="bcss", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--n_class", default=2, type=int)

    args = parser.parse_args()
    print(args)
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51] # TE 0
        palette[3:6] = [0,255,0] # NEC
        palette[6:9] = [65,105,225] # LYM
        palette[9:12] = [255,165,0] # TAS 1
        palette[12:15] = [255, 255, 255] # BACK
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0] # TUM 0 
        palette[3:6] = [0,255,0] # STR 1
        palette[6:9] = [0,0,255] # LYM
        palette[9:12] = [153, 0, 255] # NEC
        palette[12:15] = [255, 255, 255] # OTR
    elif args.dataset == 'wsss':
        palette = [0]*12
        palette[0:3] = [0, 64, 128] # Tumor 0
        palette[3:6] = [64, 128, 0] # Stroma 1
        palette[6:9] = [243, 152, 0] # Normal
        palette[9:12] = [255, 255, 255] # BACK
    elif args.dataset == 'dg':
        palette = [0]*9
        palette[0:3] = [255, 0, 0] # TUM 0 
        palette[3:6] = [0,255,0] # STR 1
        palette[6:9] = [255,255,255] # other
    PMpath = os.path.join(args.dataroot,'train_PM')
    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    model.load_state_dict(torch.load(args.weights), strict=False)
    model.eval()
    model.cuda()
    ##
    fm = 'b4_5'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    ##
    fm = 'b5_2'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    #
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)