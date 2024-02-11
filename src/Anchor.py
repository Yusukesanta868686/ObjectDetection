import torch

class AnchorGenerator:
    '''
    検出の基準となるアンカーボックスを生成するクラス
    levels: 入力特徴マップの階層
    '''

    def __init__(self, levels):
        #用意するアンカーボックスのアスペクト比(ハイパーパラメータ)
        ratios = torch.tensor([0.5, 1.0, 2.0])

        #用意するアンカーボックスの基準となる大きさに対するスケール(ハイパーパラメータ)
        scales = torch.tensor([2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)])

        #1つのアスペクト比に対して全スケールのアンカーボックスを用意するので、アンカーボックスの数は
        #アスペクト比の数 * スケール数になる
        self.num_anchors = ratios.shape[0] * scales.shape[0]

        #各階層の特徴マップでの1画素の移動量が入力画像での何画素の移動になるかを表す数値
        self.strides = [2 ** level for level in levels]

        self.anchors = []

        for level in levels:
            #現階層における基準となる正方形のアンカーボックスの1辺の長さ
            #深い階層のアンカーボックスには大きい物体の検出を担当させるため、基準の長さを長く設定
            base_length = 2 ** (level + 2)

            #アンカーボックスの1辺の長さをスケール
            scaled_lengths = base_length * scales

            #アンカーボックスが正方形の場合の面積を計算
            anchor_areas = scaled_lengths ** 2

            #アスペクト比に応じて辺の長さを変更
            #w * h = w * w * ratio = area より w = (area / ratio) ** 0.5
            #unsqueezeとブロードキャストによりアスペクト比の数 * スケール数の数のアンカーボックスの幅と高さを生成
            anchor_widths = (anchor_areas / ratios.unsqueeze(1)) ** 0.5
            anchor_heights = anchor_widths * (ratios.unsqueeze(1))

            #アスペクト比の数 * スケール数の行列を平坦化
            anchor_widths = anchor_widths.flatten()
            anchor_heights = anchor_heights.flatten()

            #アンカーボックスの中心を原点とした時のxmin, ymin, xmax, ymaxのオフセットを計算
            anchor_xmin = -0.5 * anchor_widths
            anchor_ymin = -0.5 * anchor_heights
            anchor_xmax = 0.5 * anchor_widths
            anchor_ymax = 0.5 * anchor_heights

            level_anchors = torch.stack(
                (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim = 1
            )

            self.anchors.append(level_anchors)

    '''
    アンカーボックス生成関数
    feature_sizes: 入力される複数の特徴マップそれぞれの大きさ
    '''

    #関数内で勾配を計算する必要がないことを明示
    @torch.no_grad()
    def generate(self, feature_sizes):
        #各階層の特徴マップの大きさを入力として、特徴マップの各画素におけるアンカーボックスを生成
        anchors = []
        for stride, level_anchors, feature_size in zip(self.strides, self.anchors, feature_sizes):
            #現階層の特徴マップの大きさ
            height, width = feature_size

            #入力画像の画素の移動量を表すstrideを使って特徴マップの画素の位置 -> 入力画像の画素の位置に変換
            #画素の中心位置を計算するために0.5を加算
            xs = (torch.arange(width) + 0.5) * stride
            ys = (torch.arange(height) + 0.5) * stride

            #x座標のリストとy座標のリストを組み合わせてグリッド上の座標を生成(meshgrid関数の役割)
            grid_x, grid_y = torch.meshgrid(xs, ys, indexing = 'xy')

            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            #各画素の中心位置にアンカーボックスのxmin, ymin, xmax, ymaxのオフセットを加算
            anchor_xmin = (grid_x.unsqueeze(1) + level_anchors[:, 0]).flatten()
            anchor_ymin = (grid_y.unsqueeze(1) + level_anchors[:, 1]).flatten()
            anchor_xmax = (grid_x.unsqueeze(1) + level_anchors[:, 2]).flatten()
            anchor_ymax = (grid_y.unsqueeze(1) + level_anchors[:, 3]).flatten()

            #第一軸を追加してxmin, ymin, xmax, ymaxを連結
            level_anchors = torch.stack((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), dim = 1)

            anchors.append(level_anchors)

        #全階層のアンカーボックスを連結
        anchors = torch.cat(anchors)

        return anchors