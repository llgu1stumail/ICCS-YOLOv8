import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml/weights/best.pt')
    model.val(data='/media/db306/e835eb80-e8ad-4728-8263-61e52cc524a1/gll/ICCS-YOLOv8/ultralytics-main/dataset/data-slab.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='3yolov8-C2f-SMPCGLU-CSP-PTB-ReCalibraP2345-LSCD-G',
              )
   # python3 val.py