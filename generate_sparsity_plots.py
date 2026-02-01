import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ultralytics import YOLO
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 配置参数
# WEIGHTS_DIR = r"d:\Python\Python Project\YOLOPrune\ultralytics\PruneDemo\runs\detect\visdrone_Constraint\weights"
# OUTPUT_DIR = r"d:\Python\Python Project\YOLOPrune\ultralytics\PruneDemo\sparsity_analysis_plots"

WEIGHTS_DIR = r"D:\Python\Python Project\yolov11-prune\runs\train-sparsity-1080\weights"
OUTPUT_DIR = r"D:\Python\Python Project\yolov11-prune\runs\sparsity_analysis_plots-1080"
SKIP_FIRST_BN = True
BN_THRESHOLD = 0.1

# 创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从权重文件中动态提取训练参数
def extract_training_params(weight_file):
    """
    从权重文件中提取训练参数
    Args:
        weight_file: 权重文件路径
    Returns:
        字典：训练参数
    """
    try:
        # 移除weights_only=True，使用安全的方式加载
        weight_dict = torch.load(weight_file, map_location='cpu')
        params = {}
        
        if 'train_args' in weight_dict:
            train_args = weight_dict['train_args']
            # 提取基本训练参数
            params['epochs'] = train_args.get('epochs', 50)
            params['lr0'] = train_args.get('lr0', 0.005)
            params['lrf'] = train_args.get('lrf', 0.1)
            params['warmup_epochs'] = train_args.get('warmup_epochs', 3)
            params['weight_decay'] = train_args.get('weight_decay', 0.0005)
            params['momentum'] = train_args.get('momentum', 0.9)
        else:
            # 默认参数
            params['epochs'] = 50
            params['lr0'] = 0.005
            params['lrf'] = 0.1
            params['warmup_epochs'] = 3
            params['weight_decay'] = 0.0005
            params['momentum'] = 0.9
        
        # 固定参数（原脚本中使用的）
        params['lr_decay_gamma'] = 0.9
        params['lr_step_size'] = 5
        params['l1_constant'] = 0.025
        
        print(f"✅ 从权重文件提取训练参数：")
        for k, v in params.items():
            print(f"   {k}: {v}")
        
        return params
    except Exception as e:
        print(f"⚠️  提取训练参数失败：{str(e)[:100]}...，使用默认值")
        return {
            'epochs': 50,
            'lr0': 0.005,
            'lrf': 0.1,
            'warmup_epochs': 3,
            'weight_decay': 0.0005,
            'momentum': 0.9,
            'lr_decay_gamma': 0.9,
            'lr_step_size': 5,
            'l1_constant': 0.025
        }

# 从权重文件夹中获取最大epoch数
def get_max_epoch(weights_dir):
    """
    从权重文件夹中获取最大epoch数
    Args:
        weights_dir: 权重文件夹路径
    Returns:
        int: 最大epoch数
    """
    epoch_files = [f for f in os.listdir(weights_dir) if f.startswith('epoch') and f.endswith('.pt')]
    epochs = []
    for f in epoch_files:
        try:
            epoch = int(f.split('epoch')[1].split('.pt')[0])
            epochs.append(epoch)
        except:
            continue
    return max(epochs) if epochs else 50

# 提取训练参数（使用best.pt作为参考）
best_weight_file = os.path.join(WEIGHTS_DIR, "best.pt")
if not os.path.exists(best_weight_file):
    # 如果best.pt不存在，使用第一个epoch的权重文件
    epoch_files = sorted([f for f in os.listdir(WEIGHTS_DIR) if f.startswith('epoch') and f.endswith('.pt')])
    if epoch_files:
        best_weight_file = os.path.join(WEIGHTS_DIR, epoch_files[0])

TRAIN_PARAMS = extract_training_params(best_weight_file)

# 从权重文件夹中获取实际的最大epoch数
ACTUAL_MAX_EPOCH = get_max_epoch(WEIGHTS_DIR)

# 动态设置参数
EPOCHS = ACTUAL_MAX_EPOCH  # 使用实际存在的最大epoch数
LR0 = TRAIN_PARAMS['lr0']
LR_DECAY_GAMMA = TRAIN_PARAMS['lr_decay_gamma']
LR_STEP_SIZE = TRAIN_PARAMS['lr_step_size']
WARMUP_EPOCHS = TRAIN_PARAMS['warmup_epochs']
L1_CONSTANT = TRAIN_PARAMS['l1_constant']

print(f"✅ 训练参数设置完成：")
print(f"   实际最大epoch数：{EPOCHS}")
print(f"   初始学习率：{LR0}")
print(f"   学习率衰减系数：{LR_DECAY_GAMMA}")
print(f"   学习率衰减步长：{LR_STEP_SIZE}")
print(f"   预热轮数：{WARMUP_EPOCHS}")
print(f"   L1正则系数：{L1_CONSTANT}")


def extract_bn_gammas(model_path):
    """
    提取模型的BN层gamma值，跳过第一个BN层
    Args:
        model_path: 模型权重路径
    Returns:
        numpy数组：BN层gamma值的绝对值
    """
    model = YOLO(model_path)
    model = model.model
    
    # 解并行处理
    if hasattr(model, 'module'):
        model = model.module
    
    bn_gammas = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            if SKIP_FIRST_BN and "model.0." in name:
                continue
            bn_gammas.extend(layer.weight.data.abs().cpu().numpy())
    return np.array(bn_gammas)


def load_epoch_weights(weights_dir, start_epoch=0, end_epoch=50):
    """
    加载指定epoch范围内的权重文件
    Args:
        weights_dir: 权重文件夹路径
        start_epoch: 起始epoch
        end_epoch: 结束epoch
    Returns:
        字典：{epoch: gamma_values}
    """
    gamma_history = {}
    
    for epoch in range(start_epoch, end_epoch + 1):
        weight_file = os.path.join(weights_dir, f'epoch{epoch}.pt')
        if os.path.exists(weight_file):
            gammas = extract_bn_gammas(weight_file)
            gamma_history[epoch] = gammas
            print(f"✅ 提取Epoch {epoch} 的BN γ值：共{len(gammas)}个")
        else:
            print(f"⚠️  未找到Epoch {epoch} 的权重文件：{weight_file}")
    
    return gamma_history


def generate_3d_gamma_plot():
    """
    生成real_3d_gamma_metrics.png
    3D折线+填充图，Z轴为真实权重个数
    """
    print("\n" + "="*60)
    print("📊 开始生成3D BN γ分布图表")
    print("="*60)
    
    # 加载所有epoch的gamma值
    gamma_history = load_epoch_weights(WEIGHTS_DIR, end_epoch=EPOCHS)
    
    if not gamma_history:
        print("❌ 未找到任何有效权重文件")
        return
    
    # 准备数据
    sorted_epochs = sorted(gamma_history.keys())
    valid_epochs = sorted_epochs
    valid_gamma_data = [gamma_history[epoch] for epoch in valid_epochs]
    
    if len(valid_gamma_data) == 0:
        print("❌ 所有轮次的BN γ数据均为空或无效")
        return
    
    # 全局配置
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    
    # 创建图表（只保留3D图，调整大小）
    fig = plt.figure(figsize=(16, 12))
    
    # 子图：3D折线+填充图
    ax2 = fig.add_subplot(111, projection='3d')
    
    # 预处理γ数据：确定X轴范围和区间
    all_gamma = [g for gamma_list in valid_gamma_data for g in gamma_list]
    min_g = 0
    max_g = np.max(all_gamma) if len(all_gamma) > 0 else 1.0
    num_bins = 100
    gamma_bins = np.linspace(min_g, max_g, num_bins + 1)
    gamma_x = (gamma_bins[:-1] + gamma_bins[1:]) / 2
    
    # 选取部分轮次（最多显示20轮）
    step = max(1, len(valid_epochs) // 20)
    select_epochs = valid_epochs[::step]
    select_gamma = valid_gamma_data[::step]
    
    # 计算每轮γ的真实个数
    z_count = []
    for gamma_list in select_gamma:
        try:
            count, _ = np.histogram(gamma_list, bins=gamma_bins)
            z_count.append(count)
        except Exception as e:
            print(f"⚠️  第{select_epochs[len(z_count)]}轮计数失败：{e}，跳过该轮")
            continue
    
    if len(z_count) == 0:
        ax2.text(0.5, 0.5, 0.5, "Count calculation failed for all epochs", 
                 ha='center', va='center', transform=ax2.transAxes, 
                 fontsize=12, color='red')
        print("❌ 所有轮次的个数计算均失败")
    else:
        z_count = np.array(z_count)
        select_epochs = select_epochs[:len(z_count)]
        
        # 创建渐变颜色映射
        cmap = plt.get_cmap('viridis')
        fill_colors = [cmap(i/len(select_epochs)) for i in range(len(select_epochs))]
        
        # 遍历每一轮：绘制折线 + 填充折线下方面积
        for i, epoch in enumerate(select_epochs):
            # 绘制基础3D折线
            ax2.plot(gamma_x, [epoch]*len(gamma_x), z_count[i], 
                    color='black', alpha=0.8, linewidth=1)
            
            # 构造折线下方的填充区域
            x_coords = np.concatenate([gamma_x, gamma_x[::-1]])
            y_coords = np.concatenate([[epoch]*len(gamma_x), [epoch]*len(gamma_x)])
            z_coords = np.concatenate([z_count[i], np.zeros_like(z_count[i])])
            
            # 构造3D多边形顶点
            verts = [list(zip(x_coords, y_coords, z_coords))]
            # 创建3D填充多边形
            poly = Poly3DCollection(verts, alpha=0.4)
            poly.set_facecolor(fill_colors[i])
            poly.set_edgecolor('none')
            ax2.add_collection3d(poly)
        
        # 轴标签和样式
        ax2.set_xlabel('BN-γ', fontsize=12, labelpad=10)
        ax2.set_ylabel('epochs', fontsize=12, labelpad=10)
        ax2.set_zlabel('Number of Weights', fontsize=12, labelpad=10)
        
        # 轴范围
        ax2.set_xlim(max_g, min_g)  # X轴反转
        ax2.set_ylim(min(select_epochs), max(select_epochs))
        ax2.set_zlim(0, np.max(z_count)*1.1)
        
        # 视角
        ax2.view_init(elev=20, azim=90)
        
        # 标题
        ax2.set_title('BN-γ Value', fontsize=14, pad=20)
        
        # 网格
        ax2.grid(True, alpha=0.3)
    
    # 保存图表
    save_path = os.path.join(OUTPUT_DIR, "real_3d_gamma_metrics.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n✅ 3D BN γ分布图表已保存：{save_path}")
    print(f"   📍 有效γ数据轮次：{len(valid_epochs)}/{EPOCHS+1}")
    print(f"   📍 绘图轮次（步长{step}）：{len(select_epochs)}")


def generate_bn_comparison_plot():
    """
    生成bn_weights_comparison.png
    训练前后BN权重分布对比图
    """
    print("\n" + "="*60)
    print("📊 开始生成BN权重分布对比图")
    print("="*60)
    
    # 加载初始模型和最终模型的gamma值
    # 假设best.pt是最终模型，epoch0.pt是初始模型
    initial_model_path = os.path.join(WEIGHTS_DIR, "epoch0.pt")
    final_model_path = os.path.join(WEIGHTS_DIR, "last.pt")
    
    # 如果best.pt不存在，使用最后一个epoch的模型
    if not os.path.exists(final_model_path):
        sorted_epochs = sorted([int(f.split('epoch')[1].split('.pt')[0]) for f in os.listdir(WEIGHTS_DIR) 
                              if f.startswith('epoch') and f.endswith('.pt')])
        if sorted_epochs:
            final_epoch = sorted_epochs[-1]
            final_model_path = os.path.join(WEIGHTS_DIR, f"epoch{final_epoch}.pt")
            print(f"🔄 使用最后一个epoch的模型：epoch{final_epoch}.pt")
    
    if not os.path.exists(initial_model_path):
        print(f"❌ 未找到初始模型：{initial_model_path}")
        return
    
    if not os.path.exists(final_model_path):
        print(f"❌ 未找到最终模型：{final_model_path}")
        return
    
    # 提取gamma值
    initial_gammas = extract_bn_gammas(initial_model_path)
    final_gammas = extract_bn_gammas(final_model_path)
    
    # 全局配置
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    
    # 绘制训练前分布
    ax.hist(initial_gammas, bins=50, alpha=0.7, color='#3498db', 
            label=f'Pre-training (Mean: {initial_gammas.mean():.4f})', edgecolor='black')
    # 绘制训练后分布
    ax.hist(final_gammas, bins=50, alpha=0.7, color='#e74c3c', 
            label=f'Post-sparsity Training (Mean: {final_gammas.mean():.4f})', edgecolor='black')
    
    # 添加稀疏阈值线
    ax.axvline(x=BN_THRESHOLD, color='#9b59b6', linestyle='--', linewidth=2, 
               label=f'Sparsity Threshold: {BN_THRESHOLD}')
    
    # 标注关键统计信息
    ax.text(0.02, 0.98, 
            f'Pre-training:\n'\
            f'  Total Weights: {len(initial_gammas)}\n'\
            f'  < Threshold: {(initial_gammas < BN_THRESHOLD).sum()} ({(initial_gammas < BN_THRESHOLD).sum()/len(initial_gammas):.2%})\n'\
            f'Post-training:\n'\
            f'  Total Weights: {len(final_gammas)}\n'\
            f'  < Threshold: {(final_gammas < BN_THRESHOLD).sum()} ({(final_gammas < BN_THRESHOLD).sum()/len(final_gammas):.2%})',
            transform=ax.transAxes, va='top', ha='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 图表样式设置
    ax.set_xlabel('Absolute BN Weight Value', fontsize=12)
    ax.set_ylabel('Number of Weights', fontsize=12)
    ax.set_title('BN Weight Distribution: Pre-training vs Post-sparsity Training', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # 保存图片
    save_path = os.path.join(OUTPUT_DIR, "bn_weights_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 打印统计信息
    print(f"\n📊 BN权重对比统计：")
    print(f"   预训练模型：均值={initial_gammas.mean():.4f} | 稀疏数={(initial_gammas < BN_THRESHOLD).sum()} | 稀疏率={(initial_gammas < BN_THRESHOLD).sum()/len(initial_gammas):.2%}")
    print(f"   训练后模型：均值={final_gammas.mean():.4f} | 稀疏数={(final_gammas < BN_THRESHOLD).sum()} | 稀疏率={(final_gammas < BN_THRESHOLD).sum()/len(final_gammas):.2%}")
    print(f"✅ BN权重对比图已保存：{save_path}")


def generate_sparsity_results():
    """
    生成sparsity_results.png
    2x2子图布局，与原脚本完全一致
    """
    print("\n" + "="*60)
    print("📊 开始生成稀疏训练结果图")
    print("="*60)
    
    # 加载所有epoch的gamma值
    gamma_history = load_epoch_weights(WEIGHTS_DIR, end_epoch=EPOCHS)
    
    if not gamma_history:
        print("❌ 未找到任何有效权重文件")
        return
    
    # 准备数据
    sorted_epochs = sorted(gamma_history.keys())
    epochs = [epoch + 1 for epoch in sorted_epochs]  # 显示从1开始
    
    sparsity_list = []
    l1_coeff_list = []
    lr_list = []
    
    for epoch in sorted_epochs:
        # L1系数（恒定值）
        l1_coeff = L1_CONSTANT
        l1_coeff_list.append(l1_coeff)
        
        # 学习率计算
        if epoch < WARMUP_EPOCHS:
            lr = LR0 * (epoch + 1) / WARMUP_EPOCHS
        else:
            lr = LR0 * (LR_DECAY_GAMMA) ** ((epoch - WARMUP_EPOCHS) // LR_STEP_SIZE + 1)
        lr_list.append(lr)
        
        # 稀疏率
        bn_weights = gamma_history[epoch]
        sparsity = (bn_weights < BN_THRESHOLD).sum() / len(bn_weights) if len(bn_weights) > 0 else 0.0
        sparsity_list.append(sparsity)
    
    # 最终轮数据（使用最后一个epoch的数据）
    final_epoch = sorted_epochs[-1]
    final_bn_weights = gamma_history[final_epoch]
    final_sparsity = (final_bn_weights < BN_THRESHOLD).sum() / len(final_bn_weights)
    
    # 全局配置
    plt.switch_backend('Agg')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # 创建2x2子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # 子图1: BN Sparsity
    ax1.plot(epochs, sparsity_list, marker='o', color='#e74c3c', linewidth=2, label='Sparsity Ratio')
    ax1.axhline(y=final_sparsity, color='#e74c3c', linestyle='--', label=f'Final: {final_sparsity:.2%}')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Sparsity Ratio')
    ax1.set_title('BN Layer Sparsity vs Training Epochs')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2: L1 Coefficient & Learning Rate
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(epochs, l1_coeff_list, marker='s', color='#3498db', linewidth=2, label='L1 Coefficient')
    line2 = ax2_twin.plot(epochs, lr_list, marker='^', color='#2ecc71', linewidth=2, label='Learning Rate')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('L1 Coefficient', color='#3498db')
    ax2_twin.set_ylabel('Learning Rate', color='#2ecc71')
    ax2.set_title('L1 Coefficient & Learning Rate')
    ax2.legend(line1 + line2, [l.get_label() for l in line1 + line2], loc='upper left')
    ax2.grid(alpha=0.3)
    
    # 子图3: Final BN Weight Distribution
    ax3.hist(final_bn_weights, bins=50, color='#f39c12', alpha=0.7, edgecolor='black')
    ax3.axvline(x=BN_THRESHOLD, color='#e74c3c', linestyle='--', label=f'Sparsity Threshold: {BN_THRESHOLD}')
    ax3.set_xlabel('Absolute BN Weight Value')
    ax3.set_ylabel('Number of Weights')
    ax3.set_title(f'Final BN Weight Distribution (Total: {len(final_bn_weights)})')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 子图4: Key Metrics Summary
    metrics = ['Final Sparsity', 'Final L1 Coeff', 'Final LR', 'BN Weight Mean']
    values = [
        final_sparsity,
        L1_CONSTANT,
        lr_list[-1],
        np.mean(final_bn_weights)
    ]
    ax4.bar(metrics, values, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'])
    ax4.set_ylabel('Value')
    ax4.set_title('Sparsity Training Key Metrics')
    # 给每个柱子加数值标签
    for i, v in enumerate(values):
        ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    ax4.tick_params(axis='x', rotation=15)
    ax4.grid(alpha=0.3)
    
    # 保存图表
    save_path = os.path.join(OUTPUT_DIR, "sparsity_results.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ 稀疏训练结果图已保存：{save_path}")


def main():
    """
    主函数，生成所有三幅图
    """
    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 生成三幅图
    generate_3d_gamma_plot()
    generate_bn_comparison_plot()
    generate_sparsity_results()
    
    print("\n" + "="*60)
    print("✅ 所有图表生成完成！")
    print(f"📁 输出文件夹：{OUTPUT_DIR}")
    print("包含以下文件：")
    print("   - real_3d_gamma_metrics.png")
    print("   - bn_weights_comparison.png")
    print("   - sparsity_results.png")
    print("="*60)


if __name__ == "__main__":
    main()
