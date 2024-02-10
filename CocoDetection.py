import torch
import torchvision
import numpy as np

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    img_directory: 画像ファイルが保存されているディレクトリパスへのパス
    anno_file: アノテーションファイルのパス
    transform: データ拡張と整形を行うクラスインスタンス
    """

    def __init__(self, img_directory, anno_file, transform = None):
        super().__init__(img_directory, anno_file)

        self.transform = transform

        #カテゴリIDに欠番があるため、それを埋めてクラスIDに割り当て
        self.classes = []

        #もともとのクラスIDと新しく割り当てたクラスIDを相互に変換する
        self.coco_to_pred = {}
        self.pred_to_coco = {}

        for i, category_id in enumerate(sorted(self.coco.cats.keys())): #Cocoデータセット内の全てのカテゴリのIDが取得される→sort
            self.classes.append(self.coco.cats[category_id]["name"])
            self.coco_to_pred[category_id] = i
            self.pred_to_coco[i] = category_id

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        #親クラスのコンストラクタでself.idsに画像IDが格納されているのでそれを取得
        img_id = self.ids[idx]

        #物体の集合を一つの矩形でアノテーションしているものを除外
        target = [obj for obj in target if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        #学習用に当該画像に映る物体のクラスIDと矩形を取得
        #クラスIDはコンストラクタで新規に割り当てられたIDに変換
        classes = torch.tensor([self.coco_to_pred[obj['category_id']] for obj in target], dtype = torch.int64)
        boxes = torch.tensor([obj['bbox'] for obj in target], dtype = torch.float32)

        #矩形が0個の時、boxes.shape == [0]となってしまうため、第一軸に4を追加して軸数と第二軸の次元を合わせる
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4))

        width, height = img.size
        #xyhw -> xyxy
        boxes[:, 2:] += boxes[:, :2]

        #矩形が画像領域内に収まるように値をクリップ
        boxes[:, ::2] = boxes[:, ::2].clamp(min = 0, max = width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min = 0, max = height)

        #学習のための正解データを用意
        #クラスIDや矩形など渡すものが多岐にわたるため、辞書で用意
        target = {
            'iamge_id': torch.tensor(img_id, dtype = torch.int64),
            'classes': classes,
            'boxes': boxes,
            'size': torch.tensor((width, height), dtype = torch.int64),
            'orig_size': torch.tensor((width, height), dtype = torch.int64),
            'orig_img': torch.tensor(np.array(img))
        }

        #データ拡張と変形
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target
    
    def to_coco_label(self, label: int):
        #モデルで予測されたクラスIDからCOCOのクラスIDに変換する関数
        #label: 予測されたクラスID

        return self.pred_to_coco[label]