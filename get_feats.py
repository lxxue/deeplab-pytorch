import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import click
from mmcv.utils import ProgressBar
import os
import cv2
import numpy as np

from pycocotools.coco import COCO
from mymodel import MyDeepLabV2_NOASPP_ResNet101, MyDeepLabV2_ASPP_ResNet101

class MyCOCO(Dataset):
    
    def __init__(self, data_dir, imgs):
        self.data_dir = data_dir
        self.imgs = imgs
        # R: 122.675
        # G: 116.669
        # B: 104.008
        self.mean_bgr = np.array([104.008, 116.669, 122.675])
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.imgs[idx]['file_name'])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (321, 321), interpolation=cv2.INTER_LINEAR)
        # Mean subtraction
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32)
        
    def __len__(self):
        return len(self.imgs)

@click.command()
@click.argument('model_type')
@click.argument('train_fname')
@click.argument('val_fname')
def main(model_type, train_fname, val_fname):
    print(model_type, train_fname, val_fname) 
    val_dataDir='/mnt/coco14'
    val_dataType='val2014'
    val_annFile='{}/annotations/instances_{}.json'.format(val_dataDir, val_dataType)
    # initialize COCO api for instance annotations
    val_coco=COCO(val_annFile)
    val_ids = val_coco.getImgIds()
    val_imgs = val_coco.loadImgs(val_ids)
    print(len(val_imgs))
    train_dataDir='/mnt/coco14'
    train_dataType='train2014'
    train_annFile='{}/annotations/instances_{}.json'.format(train_dataDir,train_dataType)
    # initialize COCO api for instance annotations
    train_coco=COCO(train_annFile)
    train_ids = train_coco.getImgIds()
    train_imgs = train_coco.loadImgs(train_ids)
    print(len(train_imgs))

    train_mycoco = MyCOCO("/mnt/coco14/train2014", train_imgs)
    val_mycoco = MyCOCO("/mnt/coco14/val2014", val_imgs[:100])

    batch_size = 6

    train_dataloader = DataLoader(train_mycoco, 
				  batch_size=batch_size, 
				  shuffle=False, 
				  num_workers=batch_size, 
				  pin_memory=True, 
				  drop_last=False)
    val_dataloader = DataLoader(val_mycoco, 
				batch_size=batch_size, 
				shuffle=False, 
				num_workers=batch_size, 
				pin_memory=True, 
				drop_last=False)

    print(model_type)
    if model_type == "aspp":
        model = MyDeepLabV2_ASPP_ResNet101(n_classes=182)
        print(model.load_state_dict(torch.load("ckpt/deeplabv2_resnet101_msc-cocostuff164k-100000.pth")))
        train_feats = np.zeros((len(train_mycoco), 182, 10, 10), dtype=np.float32) 
        val_feats = np.zeros((len(val_mycoco), 182, 10, 10), dtype=np.float32) 
    elif model_type == "noaspp":
        model = MyDeepLabV2_NOASPP_ResNet101(n_classes=182)
        print(model.load_state_dict(torch.load("ckpt/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"), strict=False))
        train_feats = np.zeros((len(train_mycoco), 2048, 5, 5), dtype=np.float32) 
        val_feats = np.zeros((len(val_mycoco), 2048, 5, 5), dtype=np.float32) 
    else:
        print("argument must be aspp or noaspp")

    model.eval()
    model.cuda()

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    prog_bar = ProgressBar(len(train_dataloader))
    for i, data in enumerate(train_dataloader):
        train_feats[i*batch_size:(i+1)*batch_size] = \
                model(data.cuda()).cpu().numpy()
        prog_bar.update()
    prog_bar = ProgressBar(len(val_dataloader))
    for i, data in enumerate(val_dataloader):
        val_feats[i*batch_size:(i+1)*batch_size] = \
                model(data.cuda()).cpu().numpy()
        prog_bar.update()

    np.save(train_fname, train_feats)
    np.save(val_fname, val_feats)


if __name__ == "__main__":
    main()
