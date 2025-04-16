import os  # 导入操作系统模块，用于文件和路径操作
import random  # 导入随机数模块，用于生成随机数
import numpy as np  # 导入NumPy库，用于科学计算
import pandas as pd  # 导入Pandas库，用于数据分析和处理
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
from PIL import Image  # 导入PIL库的Image模块，用于图像处理
import torch  # 导入PyTorch库，深度学习框架
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器类
from torchvision import transforms, models  # 导入PyTorch视觉库的图像转换和预训练模型
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # 导入评估指标


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)  # 设置Python随机数生成器的种子
    np.random.seed(seed)  # 设置NumPy随机数生成器的种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器的种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # 设置所有PyTorch GPU随机数生成器的种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试，确保结果的确定性


set_seed(42)  # 调用设置随机种子函数，使用种子值42

# 定义数据集路径和类别
DATA_DIR = '/home/vgg16_aml_classification/aml_data'  # 设置数据集目录路径
CLASSES = ['CBFB_MYH11', 'normal', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']  # 定义数据类别名称
NUM_CLASSES = len(CLASSES)  # 计算类别数量

# 预训练模型文件名
PRETRAINED_MODEL_PATH = "vgg16-397923af.pth"  # 设置预训练模型的文件路径,这个是vgg16的预训练权重，用预训练权重效果会更好一些，这个可以换成其他的预训练权重

# 设备配置，这个是使用GPU加速训练，用的是pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有GPU可用，有则使用GPU，否则使用CPU
print(f"使用设备: {device}")  # 打印使用的设备信息


#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇数据处理部分⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
# 图像转换，原本的数据格式是tif的，用这个处理成张量
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224像素，原本的尺寸是144×144
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用均值和标准差进行标准化
    ]),
    'val': transforms.Compose([  # 验证集图像转换
        transforms.Resize((224, 224)),  # 调整图像大小为224x224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差进行标准化
    ]),
    'test': transforms.Compose([  # 测试集图像转换流程
        transforms.Resize((224, 224)),  # 调整图像大小为224x224像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet数据集的均值和标准差进行标准化
    ])
}

# 创建图像路径和标签列表，这个是加载数据集路径，输出数据的信息
def create_dataset_list(data_dir, classes):
    image_paths = []  # 初始化图像路径列表
    labels = []  # 初始化标签列表

    for class_idx, class_name in enumerate(classes):  # 遍历所有类别
        class_dir = os.path.join(data_dir, class_name)  # 构造类别目录路径
        if not os.path.isdir(class_dir):  # 检查目录是否存在
            continue  # 如果目录不存在，继续下一个类别

        class_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir)
                       if file.lower().endswith('.tif')]  # 获取该类别下所有.tif图像文件的完整路径

        print(f"类别 {class_name} 的图像数量: {len(class_files)}")  # 打印每个类别的图像数量
        image_paths.extend(class_files)  # 将当前类别的图像路径添加到总列表中
        labels.extend([class_idx] * len(class_files))  # 为每个图像添加对应的类别索引作为标签

    return image_paths, labels  # 返回所有图像路径和对应的标签


# 自定义数据集类
class AMLDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths  # 存储图像路径列表
        self.labels = labels  # 存储标签列表
        self.transform = transform  # 存储图像转换操作

    def __len__(self):
        return len(self.image_paths)  # 返回数据集中的样本数量

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # 获取指定索引的图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB格式
        label = self.labels[idx]  # 获取对应的标签

        if self.transform:  # 如果有转换操作
            image = self.transform(image)  # 应用图像转换

        return image, label  # 返回图像和标签对


# 数据划分 - 为 80%/10%/10% 的划分
def create_data_loaders(image_paths, labels, batch_size=256, test_size=0.1, val_size=0.1):
    # 首先，将数据集分为训练集和临时集
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=test_size + val_size, stratify=labels, random_state=42
    )  # 使用分层抽样将数据集分为训练集和临时集（验证集+测试集）

    # 然后，将临时集分为验证集和测试集
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size / (test_size + val_size), stratify=temp_labels, random_state=42
    )  # 进一步将临时集分为验证集和测试集，保持分层抽样

    # 打印划分信息
    print(f"训练集大小: {len(train_paths)} ({len(train_paths) / len(image_paths):.2%})")  # 打印训练集样本数及占比
    print(f"验证集大小: {len(val_paths)} ({len(val_paths) / len(image_paths):.2%})")  # 打印验证集样本数及占比
    print(f"测试集大小: {len(test_paths)} ({len(test_paths) / len(image_paths):.2%})")  # 打印测试集样本数及占比

    # 创建数据集
    train_dataset = AMLDataset(train_paths, train_labels, transform=data_transforms['train'])  # 创建训练数据集
    val_dataset = AMLDataset(val_paths, val_labels, transform=data_transforms['val'])  # 创建验证数据集
    test_dataset = AMLDataset(test_paths, test_labels, transform=data_transforms['test'])  # 创建测试数据集

    # 创建数据加载器 - 增加batch size和worker数量
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)  # 创建训练数据加载器，打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)  # 创建验证数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)  # 创建测试数据加载器

    return train_loader, val_loader, test_loader  # 返回三个数据加载器
##⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆数据处理和加载
#
# 修改1：模型构建函数
def build_model(num_classes, pretrained_model_path="inception_v3.pth"):
    # 检查本地预训练权重
    if os.path.exists(pretrained_model_path):
        model = models.inception_v3(weights=None)
        model.load_state_dict(torch.load(pretrained_model_path))
        print("加载本地Inception v3权重")
    else:
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        torch.save(model.state_dict(), pretrained_model_path)
        print("下载并保存Inception v3权重")

    # 修改分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # 处理辅助输出层
    if model.AuxLogits is not None:
        aux_in_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    # 输入尺寸要求标记
    model.transform_input = True
    return model.to(device)


##⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇模型构建
# 修改后的VGG16模型构建函数 - 添加BN层
def build_model(num_classes, pretrained_model_path=PRETRAINED_MODEL_PATH):
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
        super(LabelSmoothingLoss, self).__init__()  # 调用父类的初始化方法
        self.confidence = 1.0 - smoothing  # 计算正确标签的置信度
        self.smoothing = smoothing  # 存储平滑参数
        self.cls = classes  # 存储类别数
        self.dim = dim  # 存储维度参数

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)  # 对预测结果应用log_softmax
        with torch.no_grad():  # 不计算梯度
            # 创建平滑的标签
            true_dist = torch.zeros_like(pred)  # 创建与预测相同形状的全零张量
            true_dist.fill_(self.smoothing / (self.cls - 1))  # 填充非目标类的平滑值
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)  # 设置目标类的值为confidence
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))  # 计算并返回平滑交叉熵损失
##⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆模型构建

##⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇模型训练
# 训练函数
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = None  # 初始化最佳模型权重为None
    best_acc = 0.0  # 初始化最佳准确率为0

    # 记录训练和验证损失、准确率
    history = {
        'train_loss': [],  # 存储训练损失
        'train_acc': [],   # 存储训练准确率
        'val_loss': [],    # 存储验证损失
        'val_acc': []      # 存储验证准确率
    }

    for epoch in range(num_epochs):  # 遍历所有训练轮次
        print(f'Epoch {epoch + 1}/{num_epochs}')  # 打印当前轮次
        print('-' * 10)  # 打印分隔线

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:  # 分别进行训练和验证
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0  # 初始化累计损失
            running_corrects = 0  # 初始化累计正确预测数

            # 遍历数据
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):  # 遍历当前阶段的所有批次数据，显示进度条
                inputs = inputs.to(device)  # 将输入数据转移到指定设备
                labels = labels.to(device)  # 将标签转移到指定设备

                # 梯度归零
                optimizer.zero_grad()  # 清空之前的梯度

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):  # 仅在训练阶段计算梯度
                    outputs = model(inputs)  # 前向传播得到输出
                    _, preds = torch.max(outputs, 1)  # 获取最高概率的预测类别
                    loss = criterion(outputs, labels)  # 计算损失

                    # 如果是训练阶段，则反向传播和优化
                    if phase == 'train':
                        loss.backward()  # 反向传播计算梯度
                        optimizer.step()  # 更新模型参数

                # 统计
                running_loss += loss.item() * inputs.size(0)  # 累加批次损失
                running_corrects += torch.sum(preds == labels.data)  # 累加正确预测数

            if phase == 'train' and scheduler is not None:  # 如果是训练阶段且存在学习率调度器
                scheduler.step()  # 更新学习率

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 计算当前轮次的平均损失
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # 计算当前轮次的平均准确率

            # 记录历史
            if phase == 'train':  # 如果是训练阶段
                history['train_loss'].append(epoch_loss)  # 记录训练损失
                history['train_acc'].append(epoch_acc.item())  # 记录训练准确率
            else:  # 如果是验证阶段
                history['val_loss'].append(epoch_loss)  # 记录验证损失
                history['val_acc'].append(epoch_acc.item())  # 记录验证准确率

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')  # 打印当前阶段的损失和准确率

            # 深拷贝最佳模型
            if phase == 'val' and epoch_acc > best_acc:  # 如果是验证阶段且准确率优于之前最佳
                best_acc = epoch_acc  # 更新最佳准确率
                best_model_wts = model.state_dict().copy()  # 保存当前模型权重

        print()  # 打印空行

    # 加载最佳模型权重
    if best_model_wts:  # 如果有最佳模型权重
        model.load_state_dict(best_model_wts)  # 加载最佳模型权重

    return model, history  # 返回训练好的模型和训练历史
#⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆模型训练

#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇模型评估
# 模型评估函数
def evaluate_model(model, test_loader, criterion):
    model.eval()  # 设置模型为评估模式

    running_loss = 0.0  # 初始化累计损失
    all_preds = []  # 初始化所有预测结果列表
    all_labels = []  # 初始化所有真实标签列表

    # 不需要梯度计算
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in tqdm(test_loader, desc="Testing"):  # 遍历测试数据集，显示进度条
            inputs = inputs.to(device)  # 将输入数据转移到指定设备
            labels = labels.to(device)  # 将标签转移到指定设备

            outputs = model(inputs)  # 前向传播得到输出
            _, preds = torch.max(outputs, 1)  # 获取最高概率的预测类别
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item() * inputs.size(0)  # 累加批次损失
            all_preds.extend(preds.cpu().numpy())  # 收集预测结果
            all_labels.extend(labels.cpu().numpy())  # 收集真实标签

    test_loss = running_loss / len(test_loader.dataset)  # 计算测试集的平均损失
    test_acc = accuracy_score(all_labels, all_preds)  # 计算测试集的准确率

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')  # 打印测试损失和准确率

    # 输出分类报告
    class_report = classification_report(all_labels, all_preds, target_names=CLASSES)  # 生成详细的分类报告
    print(class_report)  # 打印分类报告

    # 混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)  # 计算混淆矩阵

    return test_loss, test_acc, class_report, conf_matrix  # 返回测试结果
#⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆模型评估


##⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇结果可视化
# 绘制训练曲线
def plot_training_history(history):
    # 设置绘图的尺寸
    plt.figure(figsize=(12, 5))  # 创建一个12x5英寸的图形

    # 绘制准确率
    plt.subplot(1, 2, 1)  # 创建子图，1行2列的第1个
    plt.plot(history['train_acc'], label='Train Accuracy', color='#FFD700', linestyle='-')  # 绘制训练准确率曲线
    plt.plot(history['val_acc'], label='Validation Accuracy', color='#00FFFF', linestyle='-')  # 绘制验证准确率曲线
    plt.title('Model Accuracy')  # 设置标题
    plt.ylabel('Accuracy')  # 设置y轴标签
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.legend()  # 显示图例

    # 绘制损失
    plt.subplot(1, 2, 2)  # 创建子图，1行2列的第2个
    plt.plot(history['train_loss'], label='Train Loss', color='#FFD700', linestyle='-')  # 绘制训练损失曲线
    plt.plot(history['val_loss'], label='Validation Loss', color='#00FFFF', linestyle='-')  # 绘制验证损失曲线
    plt.title('Model Loss')  # 设置标题
    plt.ylabel('Loss')  # 设置y轴标签
    plt.xlabel('Epoch')  # 设置x轴标签
    plt.legend()  # 显示图例

    plt.tight_layout()  # 调整子图参数，使之填充整个图像区域
    plt.savefig('training_history.png')  # 保存图像为文件
    plt.show()  # 显示图像



# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes):
    # 自定义颜色映射
    from matplotlib.colors import LinearSegmentedColormap  # 导入颜色映射相关模块
    import matplotlib.pyplot as plt  # 导入pyplot
    import numpy as np  # 导入numpy

    # 根据图像的颜色定义新的颜色映射（浅绿色到紫色）
    colors = ["#abedd8", "#46cdcf", "#3d84a8", "#48466d", "#6C49B8", "#3B0086"]  # 自定义颜色列表
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)  # 创建自定义颜色映射

    plt.figure(figsize=(10, 8))  # 创建一个10x8英寸的图形
    plt.imshow(cm, interpolation='nearest', cmap=custom_cmap)  # 显示混淆矩阵图像
    plt.title('Confusion Matrix')  # 设置标题
    plt.colorbar()  # 添加颜色条
    tick_marks = np.arange(len(classes))  # 创建刻度标记
    plt.xticks(tick_marks, classes, rotation=45)  # 设置x轴刻度标签
    plt.yticks(tick_marks, classes)  # 设置y轴刻度标签

    # 在每个单元格中添加数字
    thresh = cm.max() / 2.  # 设置颜色阈值
    for i in range(cm.shape[0]):  # 遍历行
        for j in range(cm.shape[1]):  # 遍历列
            plt.text(j, i, format(cm[i, j], '.3f') if isinstance(cm[i, j], float) else format(cm[i, j], 'd'),
                     horizontalalignment="center",  # 设置文本水平对齐方式
                     color="white" if cm[i, j] > thresh else "black")  # 根据背景颜色深浅设置文本颜色

    plt.tight_layout()  # 调整子图参数，使之填充整个图像区域
    plt.ylabel('True label')  # 设置y轴标签
    plt.xlabel('Predicted label')  # 设置x轴标签
    plt.savefig('confusion_matrix.png')  # 保存图像为文件
    plt.show()  # 显示图像
#⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆结果可视化

#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇保存模型训练的权重
# 保存模型函数
def save_model(model, save_path="aml_vgg16_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),  # 保存模型状态字典
        'classes': CLASSES  # 保存类别信息
    }, save_path)  # 将模型保存到指定路径
    print(f"模型已保存至 {save_path}")  # 打印保存信息

#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇主函数⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇主函数⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
#⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇主函数⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
# 主函数
def main():
    # 获取图像路径和标签
    print("正在收集数据...")  # 打印信息
    image_paths, labels = create_dataset_list(DATA_DIR, CLASSES)  # 创建数据集列表

    # 创建数据加载器 - 使用更新的划分比例 (80%/10%/10%)
    print("正在创建数据加载器...")  # 打印信息
    batch_size = 256  # 增大batch size以充分利用80G显存
    train_loader, val_loader, test_loader = create_data_loaders(
        image_paths, labels, batch_size=batch_size, test_size=0.1, val_size=0.1
    )  # 创建数据加载器

    dataloaders = {
        'train': train_loader,  # 训练数据加载器
        'val': val_loader  # 验证数据加载器
    }

    # 构建模型 - 使用预训练权重
    print("正在构建VGG16模型（使用预训练权重）...")  # 打印信息
    model = build_model(NUM_CLASSES)  # 构建VGG16模型

    # 使用带标签平滑的损失函数替代标准交叉熵
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=0.1)  # 创建带标签平滑的损失函数

    # 设置分层学习率 - 特征提取层使用较小的学习率
    feature_params = []  # 初始化特征提取层参数列表
    classifier_params = []  # 初始化分类器参数列表

    for name, param in model.named_parameters():  # 遍历模型的所有参数
        if 'classifier' in name and '6' in name:  # 只有最后一层分类器
            classifier_params.append(param)  # 将分类器参数添加到列表
        else:
            feature_params.append(param)  # 将特征提取层参数添加到列表

    optimizer = optim.Adam([
        {'params': feature_params, 'lr': 0.0001},  # 较小的学习率用于预训练特征提取层
        {'params': classifier_params, 'lr': 0.001}  # 较大的学习率用于新的分类层
    ])  # 创建Adam优化器，使用不同的学习率

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 创建学习率调度器，每7个epoch将学习率乘以0.1

    # 训练模型 - 由于使用预训练权重，可以减少迭代次数
    print("开始训练模型...")  # 打印信息
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs=20
    )  # 训练模型

    # 评估模型
    print("评估模型...")  # 打印信息
    test_loss, test_acc, class_report, conf_matrix = evaluate_model(
        model, test_loader, criterion
    )  # 评估模型性能

    # 绘制训练历史
    plot_training_history(history)  # 绘制训练和验证的损失和准确率曲线

    # 绘制混淆矩阵
    plot_confusion_matrix(conf_matrix, CLASSES)  # 绘制混淆矩阵

    # 保存模型
    save_model(model)  # 保存训练好的模型

    # 输出模型架构和参数数量
    total_params = sum(p.numel() for p in model.parameters())  # 计算模型总参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算可训练参数数量

    print(f"模型总参数: {total_params}")  # 打印总参数数量
    print("训练和评估完成!")  # 打印完成信息


if __name__ == "__main__":  # 如果这个脚本是作为主程序运行
    main()  # 执行主函数
