import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 指定字体文件路径
font_path = '/home/vgg16_aml_classification/SimHei.ttf'

# 检查字体文件是否存在
if not os.path.exists(font_path):
    print(f"错误: 找不到字体文件 '{font_path}'")
    exit()
else:
    print(f"字体文件存在于: {font_path}")

# 手动注册字体
fm.fontManager.addfont(font_path)

# 获取字体名称
font_name = fm.FontProperties(fname=font_path).get_name()
print(f"字体名称: {font_name}")

# 全局设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 定义数据集路径和类别
DATA_DIR = '/home/vgg16_aml_classification/aml_data'
CLASSES = ['CBFB_MYH11', 'normal', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
NUM_CLASSES = len(CLASSES)

# 预训练模型文件名
PRETRAINED_MODEL_PATH = "vgg16-397923af.pth"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 图像转换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# 创建图像路径和标签列表
def create_dataset_list(data_dir, classes):
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir)
                       if file.lower().endswith('.tif')]

        print(f"类别 {class_name} 的图像数量: {len(class_files)}")
        image_paths.extend(class_files)
        labels.extend([class_idx] * len(class_files))

    return image_paths, labels


# 自定义数据集类
class AMLDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据划分
def create_data_loaders(image_paths, labels, batch_size=256, test_size=0.1, val_size=0.1):
    # 首先，将数据集分为训练集和临时集
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=test_size + val_size, stratify=labels, random_state=42
    )

    # 然后，将临时集分为验证集和测试集
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size / (test_size + val_size), stratify=temp_labels, random_state=42
    )

    # 打印划分信息
    print(f"训练集大小: {len(train_paths)} ({len(train_paths) / len(image_paths):.2%})")
    print(f"验证集大小: {len(val_paths)} ({len(val_paths) / len(image_paths):.2%})")
    print(f"测试集大小: {len(test_paths)} ({len(test_paths) / len(image_paths):.2%})")

    # 创建数据集
    train_dataset = AMLDataset(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = AMLDataset(val_paths, val_labels, transform=data_transforms['val'])
    test_dataset = AMLDataset(test_paths, test_labels, transform=data_transforms['test'])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader


# 原始VGG16模型构建函数（无BN层）
def build_model_no_bn(num_classes, pretrained_model_path=PRETRAINED_MODEL_PATH):
    # 检查预训练模型文件是否存在
    if os.path.exists(pretrained_model_path):
        print(f"找到预训练模型: {pretrained_model_path}")
        # 使用本地预训练权重
        model = models.vgg16(weights=None)
        model.load_state_dict(torch.load(pretrained_model_path))
        print("已加载预训练权重")
    else:
        print(f"未找到预训练模型文件，正在下载...")
        # 下载预训练模型
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 保存预训练模型权重到本地
        torch.save(model.state_dict(), pretrained_model_path)
        print(f"预训练模型已下载并保存至: {pretrained_model_path}")

    # 修改分类器以适应我们的任务
    model.classifier[-1] = nn.Linear(4096, num_classes)

    # 迁移到GPU
    model = model.to(device)
    return model


# 改进的VGG16模型构建函数（添加BN层）
def build_model_with_bn(num_classes, pretrained_model_path=PRETRAINED_MODEL_PATH):
    # 检查预训练模型文件是否存在
    if os.path.exists(pretrained_model_path):
        print(f"找到预训练模型: {pretrained_model_path}")
        # 使用本地预训练权重
        model = models.vgg16(weights=None)
        model.load_state_dict(torch.load(pretrained_model_path))
        print("已加载预训练权重")
    else:
        print(f"未找到预训练模型文件，正在下载...")
        # 下载预训练模型
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # 保存预训练模型权重到本地
        torch.save(model.state_dict(), pretrained_model_path)
        print(f"预训练模型已下载并保存至: {pretrained_model_path}")

    # 重构带有BN层的特征提取器
    features_with_bn = nn.Sequential()
    old_features = model.features

    # 遍历原始特征提取器的层，并添加BN层
    idx = 0
    bn_idx = 1
    for layer in old_features:
        if isinstance(layer, nn.Conv2d):
            features_with_bn.add_module(f"conv{idx}", layer)
            features_with_bn.add_module(f"bn{bn_idx}", nn.BatchNorm2d(layer.out_channels))
            idx += 1
            bn_idx += 1
        elif isinstance(layer, nn.ReLU):
            features_with_bn.add_module(f"relu{idx}", layer)
            idx += 1
        elif isinstance(layer, nn.MaxPool2d):
            features_with_bn.add_module(f"pool{idx}", layer)
            idx += 1

    # 替换原始特征提取器
    model.features = features_with_bn

    # 修改分类器以适应我们的任务
    classifier_with_bn = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes)
    )

    # 替换原始分类器
    model.classifier = classifier_with_bn

    # 迁移到GPU
    model = model.to(device)
    return model


# 定义带标签平滑的交叉熵损失函数
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# 训练函数
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = None
    best_acc = 0.0

    # 记录训练和验证损失、准确率
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度归零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()

        print()

    # 加载最佳模型权重
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model, history


# 模型评估函数
def evaluate_model(model, test_loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # 输出分类报告
    class_report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(class_report)

    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return test_loss, test_acc, class_report, conf_matrix


# 保存模型函数
def save_model(model, model_name):
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"模型已保存为 {model_name}.pth")


# 绘制训练历史
def plot_training_history(history_no_bn, history_with_bn, save_path):
    # 创建图表目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 设置图表风格
    plt.style.use('seaborn-v0_8-white')

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 绘制准确率曲线
    ax1.plot(history_no_bn['train_acc'], color='#FFD700', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='VGG16 Train')
    ax1.plot(history_no_bn['val_acc'], color='#FFD700', marker='s', linestyle='--', linewidth=2, markersize=6,
             label='VGG16 Validation')
    ax1.plot(history_with_bn['train_acc'], color='#00FFFF', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Improved Model Train')
    ax1.plot(history_with_bn['val_acc'], color='#00FFFF', marker='s', linestyle='--', linewidth=2, markersize=6,
             label='Improved Model Validation')

    ax1.set_title('Accuracy Comparison', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right', fontsize=12)

    # 绘制损失曲线
    ax2.plot(history_no_bn['train_loss'], color='#FFD700', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='VGG16 Train')
    ax2.plot(history_no_bn['val_loss'], color='#FFD700', marker='s', linestyle='--', linewidth=2, markersize=6,
             label='VGG16 Validation')
    ax2.plot(history_with_bn['train_loss'], color='#00FFFF', marker='o', linestyle='-', linewidth=2, markersize=6,
             label='Improved Model Train')
    ax2.plot(history_with_bn['val_loss'], color='#00FFFF', marker='s', linestyle='--', linewidth=2, markersize=6,
             label='Improved Model Validation')

    ax2.set_title('Loss Comparison', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"对比曲线已保存至: {save_path}")


# 主函数
def main():
    # 获取图像路径和标签
    print("正在收集数据...")
    image_paths, labels = create_dataset_list(DATA_DIR, CLASSES)

    # 创建数据加载器 - 使用更新的划分比例 (80%/10%/10%)
    print("正在创建数据加载器...")
    batch_size = 256
    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=batch_size, test_size=0.1, val_size=0.1
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 设置训练轮次和结果保存路径
    num_epochs = 20
    results_path = '/home/vgg16_aml_classification/pictures/model_comparison_bn_nobn'
    os.makedirs(results_path, exist_ok=True)

    # 训练不带BN层的VGG16模型
    print("正在构建原始VGG16模型（不带BN层）...")
    model_no_bn = build_model_no_bn(NUM_CLASSES)

    # 使用带标签平滑的损失函数
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=0.1)

    # 设置分层学习率 - 特征提取层使用较小的学习率
    feature_params = []
    classifier_params = []

    for name, param in model_no_bn.named_parameters():
        if 'classifier' in name and '6' in name:  # 只有最后一层分类器
            classifier_params.append(param)
        else:
            feature_params.append(param)

    optimizer_no_bn = optim.Adam([
        {'params': feature_params, 'lr': 0.0001},
        {'params': classifier_params, 'lr': 0.001}
    ])

    # 学习率调度器
    scheduler_no_bn = optim.lr_scheduler.StepLR(optimizer_no_bn, step_size=7, gamma=0.1)

    # 训练不带BN层的模型
    print("开始训练原始VGG16模型...")
    model_no_bn, history_no_bn = train_model(
        model_no_bn, dataloaders, criterion, optimizer_no_bn, scheduler_no_bn, num_epochs=num_epochs
    )

    # 评估不带BN层的模型
    print("评估原始VGG16模型...")
    test_loss_no_bn, test_acc_no_bn, class_report_no_bn, conf_matrix_no_bn = evaluate_model(
        model_no_bn, test_loader, criterion
    )

    # 保存不带BN层的模型
    save_model(model_no_bn, os.path.join(results_path, "VGG16"))

    # 训练带BN层的改进VGG16模型
    print("正在构建改进的VGG16模型（带BN层）...")
    model_with_bn = build_model_with_bn(NUM_CLASSES)

    # 设置分层学习率 - 特征提取层使用较小的学习率
    feature_params_bn = []
    classifier_params_bn = []

    for name, param in model_with_bn.named_parameters():
        if 'classifier' in name and isinstance(model_with_bn.classifier[-1], nn.Linear):
            classifier_params_bn.append(param)
        else:
            feature_params_bn.append(param)

    optimizer_with_bn = optim.Adam([
        {'params': feature_params_bn, 'lr': 0.0001},
        {'params': classifier_params_bn, 'lr': 0.001}
    ])

    # 学习率调度器
    scheduler_with_bn = optim.lr_scheduler.StepLR(optimizer_with_bn, step_size=7, gamma=0.1)

    # 训练带BN层的模型
    print("开始训练改进的VGG16模型...")
    model_with_bn, history_with_bn = train_model(
        model_with_bn, dataloaders, criterion, optimizer_with_bn, scheduler_with_bn, num_epochs=num_epochs
    )

    # 评估带BN层的模型
    print("评估改进的VGG16模型...")
    test_loss_with_bn, test_acc_with_bn, class_report_with_bn, conf_matrix_with_bn = evaluate_model(
        model_with_bn, test_loader, criterion
    )

    # 保存带BN层的模型
    save_model(model_with_bn, os.path.join(results_path, "Improved model"))

    # 绘制对比曲线并保存
    print("正在绘制对比曲线...")
    plot_path = os.path.join(results_path, "comparison_plot.png")
    plot_training_history(history_no_bn, history_with_bn, plot_path)

    # 输出模型参数数量
    total_params_no_bn = sum(p.numel() for p in model_no_bn.parameters())
    trainable_params_no_bn = sum(p.numel() for p in model_no_bn.parameters() if p.requires_grad)

    total_params_with_bn = sum(p.numel() for p in model_with_bn.parameters())
    trainable_params_with_bn = sum(p.numel() for p in model_with_bn.parameters() if p.requires_grad)

    print(f"原始VGG16模型总参数: {total_params_no_bn}")
    print(f"改进VGG16模型总参数: {total_params_with_bn}")
    print("训练和评估完成!")

    # 输出测试准确率对比
    print(f"原始VGG16测试准确率: {test_acc_no_bn:.4f}")
    print(f"改进VGG16测试准确率: {test_acc_with_bn:.4f}")
    improvement = (test_acc_with_bn - test_acc_no_bn) * 100
    print(f"改进幅度: {improvement:.2f}%")


if __name__ == "__main__":
    main()
