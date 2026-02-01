from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载剪枝后的权重
    # model_path = "runs/finetune/pruned.pt"
    model_path = "runs/train-sparsity/weights/epoch0.pt"
    print(f"Loading pruned model from: {model_path}")

    # 加载模型
    model = YOLO(model_path)

    # 打印模型摘要
    print("\nModel summary:")
    model.model.info()
    
    # 使用visdrone数据集进行预测试
    print("\nPretesting with VisDrone dataset...")
    # 指定数据集配置文件
    data_path = "E:\\tools\\jupyter_project\\op_projects\\YOLOdemo\\visdrone.yaml"
    
    # 尝试加载数据集并进行简单测试
    try:
        # 运行模型验证（只运行少量批次）
        print("Running validation on VisDrone dataset...")
        # 设置batch=1和workers=0以避免多进程问题
        eval_results = model.val(data=data_path, batch=1, workers=0, imgsz=1080, max_det=1000)

    except Exception as e:
        print(f"Error during validation: {e}")


