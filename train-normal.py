import sys
import os
# -------------------------- 关键配置 --------------------------
# 替换为你本地克隆的ultralytics仓库根目录（包含ultralytics子文件夹的目录）
LOCAL_ULTRALYTICS_PATH = "D:\Python\Python Project\yolov11-prune\\ultralytics"
# --------------------------------------------------------------
# 将本地路径转为绝对路径（避免相对路径出错），并插入到sys.path最前面
sys.path.insert(0, os.path.abspath(LOCAL_ULTRALYTICS_PATH))
# 验证是否成功加载本地版本（可选，但建议保留）
import ultralytics
print(f"当前使用的ultralytics路径：{ultralytics.__file__}")


from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  



from ultralytics import YOLO
if __name__ == '__main__':
    # weight = "weights/officials/yolo11l.pt"
    # model = YOLO(weight)
    model = YOLO('yolo11n.pt')
    model.train(
        sr=0,
        data='E:\\tools\\jupyter_project\\op_projects\\YOLOdemo\\visdrone.yaml',
        cfg='ultralytics/cfg/default.yaml',
        project='.',
        name='runs/train-normal',
        epochs=1,
        batch=1,
        imgsz=1080,
        device='0',
        resume=False,
        workers=8,
        optimizer='SGD',
        lr0=1e-4,
        patience=50,
        multi_scale=True,
        label_smoothing=True
    )