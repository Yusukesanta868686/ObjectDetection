import streamlit as st
from PIL import Image
import torch
#import sys
#sys.path.append("/Users/hiratayuusuke/Desktop/deep-learning/ObjectDetection")
import Transforms as T
from Retinanet import RetinaNet, post_process
from train import collate_func
import numpy as np
from torchvision.utils import draw_bounding_boxes

# 推論に使用するモデルと重みファイルのパス
MODEL_PATH = "/Users/hiratayuusuke/Desktop/deep-learning/ObjectDetection/model/retinanet.pth"

# Streamlitアプリケーションの作成
def main():
    st.title("Object Detection App")

    # 画像をアップロードする
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
      
    if uploaded_image is not None:
        # 画像を表示
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # 画像をTensorに変換
        img_orig = Image.open(uploaded_image)

        width, height = img_orig.size
        
        transforms = T.Compose((
        T.RandomResize((608, ), max_size = 1024),
        T.ToTensor(),
        T.Normalize(mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225)),
        ))
        
        #データ整形を適用するためにダミーのラベルを作成
        target = {
            'classes': torch.zeros((0, ), dtype = torch.int64),
            'boxes': torch.zeros((0, 4), dtype = torch.float32),
            'size': torch.tensor((width, height), dtype = torch.int64),
            'orig_size': torch.tensor((width, height), dtype = torch.int64),
        }

        img, target = transforms(img_orig, target)
        image_tensor, targets = collate_func([(img, target)])
        
        # 推論モデルをロード
        model = RetinaNet(num_classes=2)  # クラス数は適切な値に変更する必要があります
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        model.eval()

        # 物体検出の実行
        with torch.no_grad():
            preds_class, preds_box, anchors = model(image_tensor)

            # 以下、物体検出結果の処理や表示
            scores, labels, boxes = post_process(
                preds_class, preds_box, anchors, targets,  # targetsは使用していないためNoneを渡します
                conf_threshold=0.6,  # 閾値は適切な値に変更する必要があります
                nms_threshold=0.5  # 閾値は適切な値に変更する必要があります
            )
            
            # 描画用の画像を用意
            img = torch.tensor(np.asarray(img_orig))
            img = img.permute(2, 0, 1)
            
            # クラスIDをクラス名に変換
            class_names = ['person', 'car']  # 検出対象のクラス名
            labels = [class_names[label] for label in labels[0]]
            
            # 矩形を描画
            img_with_boxes = draw_bounding_boxes(
                img, boxes[0], labels, colors='red',
                font_size=42, width=4
            )
            img_with_boxes = img_with_boxes.permute(1, 2, 0)

            # PIL形式に変換してからStreamlitで画像を表示
            img_with_boxes_pil = Image.fromarray(img_with_boxes.numpy())
            st.image(img_with_boxes_pil, caption='Detected Objects', use_column_width=True)

if __name__ == "__main__":
    main()
