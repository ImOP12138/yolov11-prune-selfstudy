"""
ultralytics/cfg/__init__.py中修改了增加'finetune'从overrides中pop和重赋值,防止参数检查报错
ultralytics/engine/model.py中增加了对'maskbndict'的加载
"""
from ultralytics import YOLO
if __name__ == '__main__':
    weight = "weights-pruned-1080/pruned.pt"

    model = YOLO(weight)
    # finetune设置为True
    model.train(
        data='E:\\tools\\jupyter_project\\op_projects\\YOLOdemo\\visdrone.yaml',
        cfg='ultralytics/cfg/default.yaml',
        project='.',
        name='runs/finetune-1080-2',
        epochs=80,
        batch=1,
        imgsz=1080,
        optimizer='Adam',
        # lr0=1e-4,
        finetune=True,
        device='0',
        resume=False,
        workers=8,
        multi_scale=True,
        label_smoothing=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0
    )