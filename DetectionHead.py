import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    '''
    検出ヘッド(分類や矩形の回帰に使用)
    num_channels_per_anchor: 1アンカーに必要な出力チャネル数
    num_anchors: アンカー数
    num_features: 入力及び中間特徴量のチャネル数
    '''

    def __init__(self, num_channels_per_anchor, num_anchors = 9, num_features = 256):
        super().__init__()

        self.num_anchors = num_anchors

        '''
        特徴ピラミッドネットワークの特徴マップを分類や回帰専用の特徴マップに変換するための畳み込みブロック
        '''
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(num_features, num_features,  kernel_size = 3, padding = 1),
                          nn.ReLU(inplace = True))
            for _ in range(4)
        ])

        '''
        予測結果をアンカーボックスの数だけ用意、例えば分類ヘッドの場合、
        チャネルをアンカーボックス数*物体クラス数に設定
        '''
        self.out_conv = nn.Conv2d(num_features, num_anchors * num_channels_per_anchor, kernel_size = 3, padding = 1)

    def forward(self, x: torch.Tensor):
        for i in range(4):
            x = self.conv_blocks[i](x)

        x = self.out_conv(x)

        bs, c, h, w = x.shape

        '''
        後処理に備えて予測結果を並び替え
        permute関数により、
        [バッチサイズ、チャネル数、高さ、幅] → [バッチサイズ、高さ、幅、チャネル数]
        '''
        x = x.permute(0, 2, 3, 1)

        #第一軸に全画素の予測結果を並べる
        x = x.reshape(bs, w * h * self.num_anchors, -1)

        return x
