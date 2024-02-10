import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random

class ToTensor:
    '''
    PIL画像をテンソルに変換する関数
    img: テンソルに変換する画像
    target: 物体検出用のラベルを持つ辞書
    '''

    def __call__(self, img, target):
        img = F.to_tensor(img)

        return img, target
    
class Normalize:
    '''
    画像を標準化するクラス
    mean: R, G, Bチャネルそれぞれの平均値
    std: R, G, Bチャネルそれぞれの標準偏差
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean = self.mean, std = self.std)

        return img, target

class Compose:
    '''
    データ整形・拡張をまとめて適用するためのクラス
    transforms: データ整形・拡張のクラスのインスタンスのシーケンス
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    '''
    データ整形・拡張を連続して適用する関数
    '''
    def __call__(self, img, target):
        for transform in self.transforms:
            img, target = transform(img, target)

        return img, target

class RandomSelect:
    '''
    2種類のデータ拡張を受け取り、無作為にどちらかを適用するクラス
    '''
    def __init__(self, transform1, transform2, prob = 0.5):
        self.transform1 = transform1
        self.transform2 = transform2
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            return self.transform1(img, target)
        else:
            return self.transform2(img, target)

class RandomHorizontalFlip:
    """
    無作為に水平反転するクラス
    prob: 水平変転する確率
    """

    def __init__(self, prob = 0.5):
        self.prob = prob

        '''
        img: 水平反転する画像
        target: 物体検出用のラベルを持つ辞書
        '''

    def __call__(self, img, target):
        if random.random() < self.prob:
            img = F.hflip(img)

        #正解矩形をx軸方向に反転
        #xmin, xmaxは水平変転すると大小が反転し、width - xmax, width - xminになる
        width = img.size[0]
        target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]

        return img, target
        
class RandomSizeCrop:
    '''
    無作為に画像を切り抜くクラス
    scale: 切り抜き前に対する切り抜き後の画像面積の下限と上限(配列)
    ratio: 切り抜き後の画像のアスペクト比の下限と上限(配列)
    '''

    def __init__(self, scale, ratio):
        self.scale = scale
        self.ratio = ratio

    '''
    無作為に画像を切り抜く関数
    img: 切り抜きをする画像
    target: 物体検出用のラベルを持つ辞書
    '''
    def __call__(self, img, target):
        width, height = img.size

        #切り抜く領域の左上の座標と幅および高さを取得
        #切り抜く領域はscaleとratioの下限と上限に従う
        top, left, cropped_height, cropped_width = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

        #左上の座標と幅および高さで指定した領域を切り抜き
        img = F.crop(img, top, left, cropped_height, cropped_width)

        #原点がx = left, y = topに移動し、合わせて矩形の座標も移動
        target['boxes'][:, ::2] -= left
        target['boxes'][:, 1::2] -= top

        #矩形の切り抜き後に領域外に出る場合は座標をクリップ
        target['boxes'][:, ::2] = target['boxes'][:, ::2].clamp(min = 0, max = cropped_width)
        target['boxes'][:, 1::2] = target['boxes'][:, 1::2].clamp(min = 0, max = cropped_height)

        #幅と高さが0より大きくなる(矩形の面積が0でない)矩形のみ保持
        keep = (target['boxes'][:, 2] > target['boxes'][:, 0]) & (target['boxes'][:, 3] > target['boxes'][:, 1])
        target['classes'] = target['classes'][keep]
        target['boxes'] = target['boxes'][keep]

        #切り抜き後の画像の大きさを保持
        target['size'] = torch.tensor([cropped_width, cropped_height], dtype = torch.int64)

        return img, target
    
class RandomResize:
    '''
    無作為に画像をアスペクト比を保持してリサイズするクラス
    min_sizes: 短辺の長さの候補、この中から無作為に長さを抽出
    max_size: 長辺の長さの最大値
    '''

    def __init__(self, min_sizes, max_size):
        self.min_sizes = min_sizes
        self.max_size = max_size

    '''
    リサイズ後の短辺と長辺を計算する関数
    min_side = 短辺の長さ
    max_side = 長辺の長さ
    target: 目標となる短辺の長さ
    '''

    def _get_target_size(self, min_side, max_side, target):
        max_side = int(max_side * target / min_side)
        min_side = target

        if max_side > self.max_size:
            min_side = int(min_side * self.max_size / max_side)
            max_side = self.max_size

        return min_side, max_side
    
    '''
    無作為に画像をリサイズする関数
    img: リサイズする画像
    target: 物体検出用のラベルを持つ辞書
    '''

    def __call__(self, img, target):
        min_size = random.choice(self.min_sizes)

        width, height = img.size

        if width < height:
            resized_width, resized_height = self._get_target_size(width, height, min_size)
        else:
            resized_height, resized_width = self._get_target_size(height, width, min_size)

        #指定したサイズに画像をリサイズ
        img = F.resize(img, (resized_height, resized_width))

        #正解矩形をリサイズ前後のスケールに合わせて変更
        ratio = resized_width / width
        target['boxes'] *= ratio

        #リサイズ後の画像の大きさを保持
        target['size'] = torch.tensor([resized_width, resized_height], dtype = torch.int64)

        return img, target