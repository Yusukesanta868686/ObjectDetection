import torch
import Transforms as T
from Retinanet import RetinaNet, post_process
from pathlib import Path
from PIL import Image
from train import collate_func
import numpy as np
from torchvision.utils import draw_bounding_boxes


def demo(config):
    transforms = T.Compose((
        T.RandomResize((608, ), max_size = 1024),
        T.ToTensor(),
        T.Normalize(mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)),
    ))
    
    #学習済みモデルパラメータを読み込み
    model = RetinaNet(len(config.classes))
    model.load_state_dict(torch.load(config.load_file, map_location = config.device))
    model.to(config.device)
    model.eval()
    
    for img_path in Path(config.img_directory).iterdir():
        img_orig = Image.open(img_path)
        width, height = img_orig.size
        
        #データ整形を適用するためにダミーのラベルを作成
        target = {
            'classes': torch.zeros((0, ), dtype = torch.int64),
            'boxes': torch.zeros((0, 4), dtype = torch.float32),
            'size': torch.tensor((width, height), dtype = torch.int64),
            'orig_size': torch.tensor((width, height), dtype = torch.int64),
        }
        
        #データ整形
        img, target = transforms(img_orig, target)
        imgs, targets = collate_func([(img, target)])
        
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            targets = [{k: v.to(model.get_device())
                        for k, v in target.items()}
                       for target in targets]
            
            preds_class, preds_box, anchors = model(imgs)
            
            scores, labels, boxes = post_process(
                preds_class, preds_box, anchors, targets,
                conf_threshold = config.conf_threshold,
                nms_threshold = config.nms_threshold
            )
            
            #描画用の画像を用意
            img = torch.tensor(np.asarray(img_orig))
            img = img.permute(2, 0, 1)
            
            #クラスIDをクラス名に変換
            labels = [config.classes[label] for label in labels[0]]
            
            #矩形を描画
            img = draw_bounding_boxes(
                img, boxes[0], labels, colors = 'red',
                font_size = 42, width = 4
            )
            img = img.permute(1, 2, 0)
            img = img.to('cpu').numpy()
            img = Image.fromarray(img)
            display(img)