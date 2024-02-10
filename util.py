import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch import nn
from torch.utils.data import Dataset

def convert_to_xywh(boxes: torch.tensor):
    wh = boxes[..., 2:] - boxes[..., :2] #幅と高さを求める
    xy = boxes[...,:2] + wh / 2 #中心の座標を求める　
    boxes = torch.cat((xy, wh), dim = 1)

    return boxes

def convert_to_xyxy(boxes: torch.tensor): 
    xymin = boxes[..., :2] - boxes[..., 2:] / 2
    xymax = boxes[..., 2:] - xymin
    boxes = torch.cat((xymin, xymax), dim = 1)

    return boxes

def calc_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    #積集合の左上の座標を取得
    intersect_lift_top = torch.maximum(boxes1[:, :2].unsqueeze(1), boxes2[:, :2])
    #積集合の右上の座標を取得
    intersect_right_bottom = torch.minimum(boxes1[:, 2:].unsqueeze(1), boxes2[:, 2:])

    #積集合の幅と高さを算出し、面積を計算
    intersect_width_height = (intersect_right_bottom - intersect_lift_top).clamp(min = 0)
    intersect_areas = intersect_width_height.prod(dim = 2)

    #それぞれの矩形の面積を計算　
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    #和集合の面積を計算　
    union_areas = areas1.unsqueeze(1) + areas2 - intersect_areas

    ious = intersect_areas / union_areas

    return ious, union_areas
  

'''
データセットを分割するための2つの排反なインデックス集合を生成する関数
dataset    : 分割対象のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


'''
t-SNEのプロット関数
data_loader: プロット対象のデータを読み込むデータローダ
model      : 特徴量抽出に使うモデル
num_samples: t-SNEでプロットするサンプル数
'''
def plot_t_sne(data_loader: Dataset, model: nn.Module,
               num_samples: int):
    model.eval()

    # t-SNEのためにデータを整形
    x = []
    y = []
    for imgs, labels in data_loader:
        with torch.no_grad():
            imgs = imgs.to(model.get_device())

            # 特徴量の抽出
            embeddings = model(imgs, return_embed=True)

            x.append(embeddings.to('cpu'))
            y.append(labels.clone())

    x = torch.cat(x)
    y = torch.cat(y)

    # NumPy配列に変換
    x = x.numpy()
    y = y.numpy()

    # 指定サンプル数だけ抽出
    x = x[:num_samples]
    y = y[:num_samples]

    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    # 各ラベルの色とマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']

    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
        plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1],
                    c=[cmap(i / len(data_loader.dataset.classes))],
                    marker=markers[i], s=500, alpha=0.6, label=cls)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
    plt.show()