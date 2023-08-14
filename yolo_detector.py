from ultralytics import YOLO

import os
import random
from PIL import Image
import cv2
from utils import *

class YoloDetection:
    def __init__(self, type=None):
        if type == 'seg':
            self.model = YOLO('yolov8n-seg.pt')
        else:
            self.model = YOLO('yolov8n.pt')
    
    def __call__(self, src, threshold=0.8, save=True):
        
        output = self.model(src)[0]

        fig = self.show_detection(output, threshold)

        if save:
            if not os.path.exists('./predict'):
                os.mkdir('./predict')
        fname = src.split('/')[-1]
        fig.savefig('./predict/' + fname, dpi=200,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.show()
        return output

    def show_detection(self, output, threshold=0.8):
        # convert numpy BRG to RGB
        img = output.orig_img[:, :, ::-1]

        boxes = output.boxes.xyxy
        class_ids = output.boxes.cls
        scores = output.boxes.conf
        Masks = output.masks

        # filter less conf detection
        valid_idx = scores > threshold
        boxes = boxes[valid_idx]
        class_ids = class_ids[valid_idx]
        scores = scores[valid_idx]

        if Masks:
            masks = Masks.data[valid_idx]
            segments = [Masks.xy[i] for i in range(len(valid_idx)) if valid_idx[i]]

        # create colors mapping to class_ids
        colors = ['r','g','b','y','m','c']
        random.shuffle(colors)
        colors = [colors[int(i%len(colors))] for i in class_ids]

        cls_names =[output.names[int(x)] for x in class_ids]
        labels = [f"{cls} {round(s.numpy()*100)}%"
                  for cls, s in zip(cls_names, scores)]
        
        fig, ax = plt.subplots(frameon=False, layout='tight')

        ax.imshow(img)
        ax.set_axis_off()
        ax.margins(0, 0)

        show_bboxes(ax, boxes, labels, colors)
        if Masks:
            # resize masks to orig_img size
            masks = [cv2.resize(masks[i].numpy(), 
                                (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
                    for i in range(len(masks))]

            show_masks(ax, masks, colors, segments)    
        return fig
    
    def predict(self, src, threshold):
        self.model.predict(src, save=True, conf=threshold)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float,
                        default=.8)
    parser.add_argument('-m', '--model', default='seg')
    parser.add_argument('-i', '--img', default='../MaskRCNN/images/demo.jpg')
    parser.add_argument('-s', '--save', action='store_true')

    args = parser.parse_args()

    yolo = YoloDetection(args.model)

    if args.save:
        yolo.predict(args.img, args.threshold)
        print('Prediction saved to ./runs/detect/predict/')
    else:
        yolo(args.img, args.threshold)