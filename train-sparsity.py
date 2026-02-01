"""
修改的代码:
ultralytics/nn/modules/block.py: 对C3k2增加一个C3k布尔值属性
ultralytics/engine/trainer.py: 禁用amp, 梯度裁剪, 增加梯度惩罚项系数
ultralytics/engine/model.py: 主要是将sr参数绑定到self.trainer上
ultralytics/cfg/__init__.py: 对额外参数finetune的处理, 防止DDP下报错
ultralytics/engine/model.py: 对sr, maskbndict等额外参数的处理
"""

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
    # model = YOLO("runs/train-normal/weights/best.pt")
    model = YOLO("yolo11n-bestbase.pt")

    # L1正则的惩罚项系数sr
    model.train(
        sr=1e-3,
        data="E:\\tools\\jupyter_project\\op_projects\\YOLOdemo\\visdrone.yaml",
        cfg='ultralytics/cfg/default.yaml',
        project='.',
        name='runs/train-sparsity-1080',
        device=0, # NOTE: 目前只能单卡训, DDP下多卡训不会产生稀疏效果(TODO)
        epochs=40,
        imgsz=1080,
        batch=1,
        optimizer='SGD',
        lr0=1e-3,
        patience=50 # 注意patience要比epochs大, 防止训练过早结束
    )