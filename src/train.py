import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F  
import Transforms as T
import CocoDetection as dataset
import util
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Retinanet import RetinaNet
from loss import loss_func
from Retinanet import post_process
import json
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from collections import deque
from PIL import Image
import numpy as np
        
def train_eval(config):
    # データ拡張・整形クラスの設定
    min_sizes = (480, 512, 544, 576, 608)
    train_transforms = T.Compose((
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(min_sizes, max_size=1024),
            T.Compose((
                T.RandomSizeCrop(scale=(0.8, 1.0),
                                 ratio=(0.75, 1.333)),
                T.RandomResize(min_sizes, max_size=1024),
            ))
        ),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ))
    test_transforms = T.Compose((
        # テストは短辺最大で実行
        T.RandomResize((min_sizes[-1],), max_size=1024),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ))

    # データセットの用意
    train_dataset = dataset.CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file, transform=train_transforms)
    val_dataset = dataset.CocoDetection(
        img_directory=config.img_directory,
        anno_file=config.anno_file, transform=test_transforms)

    # Subset samplerの生成
    val_set, train_set = util.generate_subset(
        train_dataset, config.val_ratio)

    print(f'学習セットのサンプル数: {len(train_set)}')
    print(f'検証セットのサンプル数: {len(val_set)}')

    # 学習時にランダムにサンプルするためのサンプラー
    train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=train_sampler,
        collate_fn=collate_func)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, sampler=val_set,
        collate_fn=collate_func)

    # RetinaNetの生成
    model = RetinaNet(len(train_dataset.classes))
    # ResNet18をImageNetの学習済みモデルで初期化
    # 最後の全結合層がないなどのモデルの改変を許容するため、strict=False
    model.backbone.load_state_dict(torch.hub.load_state_dict_from_url(
        'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
                                   strict=False)

    # モデルを指定デバイスに転送
    model.to(config.device)
    # Optimizerの生成
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # 指定したエポックで学習率を1/10に減衰するスケジューラを生成
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[config.lr_drop], gamma=0.1)

    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 移動平均計算用
            losses_class = deque()
            losses_box = deque()
            losses = deque()
            for imgs, targets in pbar:
                imgs = imgs.to(model.get_device())
                targets = [{k: v.to(model.get_device())
                            for k, v in target.items()}
                           for target in targets]

                optimizer.zero_grad()

                preds_class, preds_box, anchors = model(imgs)

                loss_class, loss_box = loss_func(
                    preds_class, preds_box, anchors, targets)
                loss = loss_class + loss_box

                loss.backward()

                # 勾配全体のL2ノルムが上限を超えるとき上限値でクリップ
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip)

                optimizer.step()

                losses_class.append(loss_class.item())
                losses_box.append(loss_box.item())
                losses.append(loss.item())
                if len(losses) > config.moving_avg:
                    losses_class.popleft()
                    losses_box.popleft()
                    losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'loss_class': torch.Tensor(
                        losses_class).mean().item(),
                    'loss_box': torch.Tensor(
                        losses_box).mean().item()})

        # スケジューラでエポック数をカウント
        scheduler.step()

        # パラメータを保存
        torch.save(model.state_dict(), config.save_file)

        # 検証
        if (epoch + 1) % config.val_interval == 0:
            evaluate(val_loader, model, loss_func)
            
            
'''
batch: CocoDetectionからサンプルした複数の画像とラベルをまとめたもの
'''
def collate_func(batch):
    #ミニバッチのなかの画像で最大の高さと幅を取得
    max_height = 0
    max_width = 0

    for img, _ in batch:
        height, width = img.shape[1:]
        max_height = max(max_height, height)
        max_width = max(max_width, width)

    #バックボーンネットワークで特徴マップの解像度を下げる時に切り捨てが起きないように入力の幅と高さを32の倍数にしておく
    #もし32の倍数でない場合、バックボーンネットワークの特徴マップと特徴ピラミッドネットワークのアップスケーリングでできた特徴マップの
    #大きさに不整合が生じ、加算できなくなる
            
    height = (max_height + 31) // 32 * 32
    width = (max_width + 31) // 32 * 32

    #画像を1つにテンソルにまとめ、ラベルはリストに集約
    imgs = batch[0][0].new_zeros((len(batch), 3, height, width))
    targets = []
    for i, (img, target) in enumerate(batch):
        height, width = img.shape[1:] 
        imgs[i, :, :height, :width] = img

        targets.append(target)

    return imgs, targets
    
'''
dataloader: 評価に使うデータを読み込むデータローダ
model: 評価対象のモデル
loss_func: 目的関数
conf_threshold: 信頼度の閾値
nms_threshold: NMSのIoU閾値
'''

def evaluate(data_loader, model, loss_func, conf_threshold, nms_threshold):
    model.eval()

    losses_classes = []
    losses_box = []
    losses = []
    preds = []
    img_ids = []

    for imgs, targets in tqdm(data_loader, desc = '[Validation]'):
        with torch.no_grad():
            imgs = imgs.to(model.get_device())

            targets = [{k: v.to(model.get_device())
                        for k, v in target.items()}
                        for target in targets]
                
            preds_class, preds_box, anchors = model(imgs)

            loss_class, loss_box = loss_func(
                preds_class, preds_box, anchors, targets
            )
            loss = loss_class + loss_box

            losses_classes.append(loss_class)
            losses_box.append(loss_box)
            losses.append(loss)

            #後処理により最終的な検出矩形を取得
            scores, labels, boxes = post_process(
                preds_class, preds_box, anchors, targets, 
                conf_threshold = conf_threshold,
                nms_threshold = nms_threshold
            )

            for img_scores, img_labels, img_boxes, img_targets in zip(scores, labels, boxes, targets):
                img_ids.append(img_targets['image_id'].item())

                #評価のためにCocoの元々の矩形表現であるxmin, ymin, width, heightに変換
                img_boxes[:, 2:] -= img_boxes[:, :2]

                for score, label, box in zip(img_scores, img_labels, img_boxes):
                    #Coco評価用のデータの保存
                    preds.append({
                        'image_id': img_targets['image_id'].item(),
                        'category_id': data_loader.dataset.to_coco_label(
                            label.item()),
                        'score': score.item(),
                        'bbox': box.to('cpu').numpy().tolist()
                    })

                    #Cocoevalクラスを使って評価するには検出結果をjsonファイルに出力する必要があるため、jsonファイルに一時保存
                    with open ('tmp.json', 'w') as f:
                        json.dump(preds, f)

                    #一時保存した検出結果をCocoクラスを使って読み込み
                    coco_results = data_loader.dataset.coco.loadRes('tmp.json')

                    #Cocoevalクラスを用いて評価
                    coco_eval = COCOeval(
                        data_loader.dataset.coco, coco_results, 'bbox'
                    )
                    coco_eval.params.imgIds = img_ids
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()