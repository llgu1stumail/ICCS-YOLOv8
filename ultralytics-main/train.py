import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')  # yaml address (/media/db306/e835eb80-e8ad-4728-8263-61e52cc524a1/gll/ICCS-YOLOv8/ultralytics-main/ultralytics/cfg/models/v8)
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/media/db306/e835eb80-e8ad-4728-8263-61e52cc524a1/gll/ICCS-YOLOv8/ultralytics-main/dataset/data-slab.yaml',  # dataset catalog
        cache=True,
        imgsz=640,
        epochs=300,
        batch=8,
        close_mosaic=10,
        workers=8,
        device='0',
        optimizer='SGD', # using SGD
        project='/media/db306/e835eb80-e8ad-4728-8263-61e52cc524a1/gll/ICCS-YOLOv8/ultralytics-main/runs/train',  # run file save address
        name='yolov8n',  # run file save name
        )
     # conda activate gll
     # python3 train.py
