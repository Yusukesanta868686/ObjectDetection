import torch
import torch.nn as nn
import torchvision
from torchvision.ops import batched_nms
from Backbone import ResNet18
from FeaturePyramid import FeaturePyramidNetwork
from Anchor import AnchorGenerator
from DetectionHead import DetectionHead
import util

class RetinaNet(nn.Module):
    '''
    RetinaNetモデル(ResNet18バックボーン)
    num_classes: 物体クラス数
    '''

    def __init__(self, num_classes):
        super().__init__()

        self.backbone = ResNet18()
        self.fpn = FeaturePyramidNetwork()
        self.anchor_generator = AnchorGenerator(self.fpn.levels)

        #分類及び矩形ヘッド
        #検出ヘッドは全て特徴マップで共有
        self.class_head = DetectionHead(
            num_channels_per_anchor = num_classes,
            num_anchors = self.anchor_generator.num_anchors
        )

        #num_channels_per_anchor = 4は
        #(x_diff, y_diff, w_diff, h_diff)を推論するため
        self.box_head = DetectionHead(
            num_channels_per_anchor = 4,
            num_anchors = self.anchor_generator.num_anchors
        )

        self._reset_parameters()

    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
            
            #分類ヘッドの出力にシグモイドを適用して各クラスの確率を出力
            #学習開始時の確率が0.01になるようにパラメータを初期化
            prior = torch.tensor(0.01)
            nn.init.zeros_(self.class_head.out_conv.weight)
            nn.init.constant_(self.class_head.out_conv.bias, -((1.0 - prior) / prior).log())

            #学習開始時のアンカーボックスの中心位置の移動が0、大きさが1倍となるように矩形ヘッドを初期化
            nn.init.zeros_(self.box_head.out_conv.weight)
            nn.init.zeros_(self.box_head.out_conv.bias)


    '''
    順伝播関数
    '''

    def forward(self, x):
        cs = self.backbone(x)
        ps = self.fpn(*cs)
        
        preds_class = torch.cat(list(map(self.class_head, ps)), dim = 1)
        preds_box = torch.cat(list(map(self.box_head, ps)), dim = 1)

        #各特徴マップの高さと幅ををリストに保持
        feature_sizes = [p.shape[2:] for p in ps]
        anchors = self.anchor_generator.generate(feature_sizes)
        anchors = anchors.to(x.device)

        return preds_class, preds_box, anchors
    
    '''
    モデルパラメータが保持されているデバイスを返す関数
    '''
    def get_device(self):
        return self.backbone.conv1.weight.device
    
'''
後処理
preds_class: 検出矩形のクラス [バッチサイズ、アンカーボックス数、物体クラス数]
preds_box: 検出矩形のアンカーボックスからの誤差 [バッチサイズ、アンカーボックス数、4(x_diff, y_diff, w_diff, h_diff)]
anchors: アンカーボックス[アンカーボックス数、4(xmin, ymin, xmax, ymax)]
targets: ラベル
conf_threshold: 信頼度の閾値
nms_threshold: NMSのIoU閾値
'''

@torch.no_grad()
def post_process(preds_class, preds_box, anchors, targets, conf_threshold = 0.5, nms_threshold = 0.5):
    batch_size = preds_class.shape[0]

    anchors_xywh = util.convert_to_xywh(anchors)

    #中心座標の予測をスケール不変にするため、予測値をアンカーボックスの大きさでスケールする
    preds_box[:, :, :2] = anchors_xywh[:, :2] + preds_box[:, :, :2] * anchors_xywh[:, 2:]
    preds_box[:, :, 2:] = preds_box[:, :, 2:].exp() * anchors_xywh[:, 2:]

    preds_box = util.convert_to_xyxy(preds_box)

    #物体クラスの予測確率をシグモイド関数で計算
    #RetinaNetでは背景クラスは存在せず、背景を表す場合は全ての物体クラスの予測確率が低くなるように実装されている
    preds_class = preds_class.sigmoid()

    #forループで画像毎に処理を実行
    scores = []
    labels = []
    boxes = []

    for img_preds_class, img_preds_box, img_targets in zip(preds_class, preds_box, targets):
        #検出矩形が画像ないに収まるように座標をクリップ
        img_preds_box[:, ::2] = img_preds_box[:, ::2].clamp(min = 0, max = img_targets['size'][0])
        img_preds_box[:, 1::2] = img_preds_box[:, 1::2].clamp(min = 0, max = img_targets['size'][1])      

        #検出矩形は入力画像の大きさに合わせたものになっているので、もともとの画像に合わせて検出矩形をスケール
        img_preds_box *= img_targets['orig_size'][0] / img_targets['size'][0] 

        #物体クラスのスコアとクラスIDを取得
        img_preds_score, img_preds_label = img_preds_class.max(dim = 1)

        #信頼度が閾値より高い検出矩形のみを残す
        keep = img_preds_score > conf_threshold
        img_preds_score = img_preds_score[keep]
        img_preds_label = img_preds_label[keep]
        img_preds_box = img_preds_box[keep]   

        #クラス毎にNMS(非最大値抑制, non_maximum supression)を適用
        keep_indices = batched_nms(img_preds_box, img_preds_score, img_preds_label, nms_threshold)

        scores.append(img_preds_score[keep_indices])
        labels.append(img_preds_label[keep_indices])
        boxes.append(img_preds_box[keep_indices])

    return scores, labels, boxes



