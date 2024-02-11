import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    '''
    特徴ピラミッドネットワーク
    num_features: 出力特徴量のチャネル数
    '''

    def __init__(self, num_features = 256):
        super().__init__()

        '''
        特徴ピラミッドネットワークから出力される階層レベル
        バックボーンネットワークの最終層の特徴マップを5段階とし、縮小方向に第6,7階層の2つの特徴マップを、
        拡大方向に第3,4階層の2つの特徴マップを生成
        '''
        #アンカーボックスを生成する時に使う
        self.levels = (3, 4, 5, 6, 7)

        #縮小方向の特徴抽出
        self.p6 = nn.Conv2d(512, num_features, kernel_size = 3, stride = 2, padding = 1)
        self.p7_relu = nn.ReLU(inplace = True)
        self.p7 = nn.Conv2d(num_features, num_features, kernel_size = 3, stride = 2, padding = 1)

        #拡大方向の特徴抽出
        self.p5_1 = nn.Conv2d(512, num_features, kernel_size = 1)
        self.p5_2 = nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1)

        self.p4_1 = nn.Conv2d(256, num_features, kernel_size = 1)
        self.p4_2 = nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1)

        self.p3_1 = nn.Conv2d(128, num_features, kernel_size = 1)
        self.p3_2 = nn.Conv2d(num_features, num_features, kernel_size = 3, padding = 1)

    '''
    順伝播関数
    c3, c4, c5: 特徴マップ [バッチサイズ、チャネル数、高さ、幅]
    '''

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor):
        #縮小方向の特徴抽出
        p6 = self.p6(c5)

        p7 = self.p7_relu(p6)
        p7 = self.p7(p7)

        #拡大方向の特徴抽出
        p5 = self.p5_1(c5)
        p5_up = F.interpolate(p5, scale_factor = 2) #特徴マップの拡大。2倍に拡大している
        p5 = self.p5_2(p5)

        p4 = self.p4_1(c4) + p5_up
        p4_up = F.interpolate(p4, scale_factor = 2)
        p4 = self.p4_2(p4)

        p3 = self.p3_1(c3) + p4_up
        p3 = self.p3_2(p3)

        return p3, p4, p5, p6, p7
