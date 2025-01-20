import torch
from ultralytics import RTDETR

if __name__ == '__main__':
    # Choose your yaml file
    model = RTDETR('/mnt/RTdetr/UAV_DETR/ultralytics/cfg/models/uavdetr-r50.yaml')
    # model = RTDETR('rtdetr-r18.yaml')/
    # model = RTDETR('ultralytics/cfg/modelss/rt-detr/rtdetr-DySample-p2.yaml')
    # model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-ASF-P2.yaml')
    model.model.eval()
    model.info(detailed=False)

    try:
        # Profile the model with a sample image size
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass

    # Fuse the model
    model.fuse()

    # After fusing, print the model parameter count
    print('After fuse:')
