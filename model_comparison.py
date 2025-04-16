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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


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
# 创建保存结果的文件夹
SAVE_DIR = '/home/vgg16_aml_classification/pictures/model_comparison'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES = ['CBFB_MYH11', 'normal', 'NPM1', 'PML_RARA', 'RUNX1_RUNX1T1']
NUM_CLASSES = len(CLASSES)

# 预训练模型路径
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


# 数据划分 - 为 80%/10%/10% 的划分
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

    return train_loader, val_loader, test_loader, (train_paths, train_labels, test_paths, test_labels)


# ==================== 模型定义 ====================

# 构建带预训练的VGG16模型
def build_vgg16(num_classes, pretrained_model_path=PRETRAINED_MODEL_PATH):
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

# 构建LeNet5模型
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 构建AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 构建ResNet50模型（不使用预训练）
def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    return model

'''
# 构建DenseNet121模型（不使用预训练）
def build_densenet121(num_classes):
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)
    model = model.to(device)
    return model
'''

# 初始化模型
def build_lenet5(num_classes):
    model = LeNet5(num_classes)
    model = model.to(device)
    return model


# 初始化AlexNet
def build_alexnet(num_classes):
    model = AlexNet(num_classes)
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


# ==================== 训练和评估函数 ====================

# 训练函数
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=20, model_name=""):
    best_model_wts = None
    best_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        history['epoch'].append(epoch + 1)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{model_name} {phase}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if isinstance(model, models.inception.Inception3) and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()

        print()

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    # 保存训练历史到CSV文件
    history_df = pd.DataFrame({
        'epoch': history['epoch'],
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })

    history_df.to_csv(os.path.join(SAVE_DIR, f"{model_name}_training_history.csv"), index=False)
    print(f"训练历史已保存到 {os.path.join(SAVE_DIR, f'{model_name}_training_history.csv')}")

    return model, history


# 模型评估函数
def evaluate_model(model, test_loader, criterion, model_name=""):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if isinstance(model, models.inception.Inception3):
                outputs = model(inputs)
            else:
                outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 计算每个类别的准确率
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)

    # 计算每个类别的准确率
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(NUM_CLASSES)]

    print(f'{model_name} Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    class_report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(class_report)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 保存评估结果到CSV文件
    result_dict = {
        'model_name': model_name,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }

    # 添加每个类别的准确率
    for i, class_name in enumerate(CLASSES):
        result_dict[f'{class_name}_accuracy'] = class_accuracies[i]

    # 创建DataFrame并保存
    result_df = pd.DataFrame([result_dict])
    result_df.to_csv(os.path.join(SAVE_DIR, f"{model_name}_evaluation_results.csv"), index=False)
    print(f"评估结果已保存到 {os.path.join(SAVE_DIR, f'{model_name}_evaluation_results.csv')}")

    return test_loss, test_acc, class_report, conf_matrix, class_accuracies


# ==================== 可视化函数 ====================
# 绘制训练曲线
def plot_training_history(histories, model_names):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if history:  # 仅绘制具有训练历史的模型（SVM没有训练历史）
            plt.plot(history['train_acc'], linestyle='-', label=f'{name} Train Acc')
            plt.plot(history['val_acc'], linestyle='--', label=f'{name} Val Acc')
    plt.title('Models Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if history:  # 仅绘制具有训练历史的模型
            plt.plot(history['train_loss'], linestyle='-', label=f'{name} Train Loss')
            plt.plot(history['val_loss'], linestyle='--', label=f'{name} Val Loss')
    plt.title('Models Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'training_history_comparison.png'))
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["#abedd8", "#46cdcf", "#3d84a8", "#48466d", "#6C49B8", "#3B0086"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=custom_cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.3f') if isinstance(cm[i, j], float) else format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(SAVE_DIR, f'{title}.png'))
    plt.show()


# 绘制模型对比结果柱状图
def plot_model_comparison_by_class(model_names, class_accuracies_list):
    """
    绘制每个模型在各个类别上的准确率对比柱状图

    参数:
        model_names: 模型名称列表
        class_accuracies_list: 每个模型的类别准确率列表
    """
    x = np.arange(len(CLASSES))  # 类别位置
    width = 0.7 / len(model_names)  # 柱状图宽度

    fig, ax = plt.figure(figsize=(14, 8)), plt.subplot(111)

    # 设置不同的颜色
    colors = ['#FF9999', '#66B2FF', '#FFCC99', '#C2C2F0']

    # 绘制每个模型的柱状图
    for i, (model_name, class_accs) in enumerate(zip(model_names, class_accuracies_list)):
        ax.bar(x + i * width - width * len(model_names) / 2 + width / 2, class_accs, width, label=model_name,
               color=colors[i % len(colors)])

    # 添加标签、标题和图例
    ax.set_ylabel('准确率', fontsize=14)
    ax.set_xlabel('类别', fontsize=14)
    ax.set_title('各模型在不同类别上的准确率对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # 设置y轴范围从0到1
    ax.set_ylim(0, 1.0)

    # 为每个柱添加数值标签
    for i, (model_name, class_accs) in enumerate(zip(model_names, class_accuracies_list)):
        for j, v in enumerate(class_accs):
            ax.text(j + i * width - width * len(model_names) / 2 + width / 2, v + 0.02,
                    f'{v:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_comparison_by_class.png'), dpi=300)
    plt.show()


# 绘制模型总体准确率对比柱状图
def plot_model_overall_accuracy(model_names, accuracies):
    """
    绘制每个模型的总体准确率对比柱状图

    参数:
        model_names: 模型名称列表
        accuracies: 每个模型的总体准确率列表
    """
    colors = ['#FF9999', '#66B2FF', '#FFCC99', '#C2C2F0']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])

    # 为每个柱添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

    plt.ylim(0, 1.0)
    plt.ylabel('准确率')
    plt.title('各模型总体准确率对比')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_overall_accuracy.png'), dpi=300)
    plt.show()


# 绘制所有模型的损失对比图
def plot_model_loss_comparison(histories, model_names):
    """
    绘制所有模型训练和验证损失对比图

    参数:
        histories: 每个模型的训练历史
        model_names: 模型名称列表
    """
    plt.figure(figsize=(12, 10))

    # 训练损失对比
    plt.subplot(2, 1, 1)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if history:  # 确保有训练历史
            plt.plot(history['train_loss'], linestyle='-', marker='o', markersize=4, label=f'{name}')
    plt.title('模型训练损失对比')
    plt.ylabel('损失值')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    # 验证损失对比
    plt.subplot(2, 1, 2)
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if history:  # 确保有训练历史
            plt.plot(history['val_loss'], linestyle='-', marker='o', markersize=4, label=f'{name}')
    plt.title('模型验证损失对比')
    plt.ylabel('损失值')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'model_loss_comparison.png'), dpi=300)
    plt.show()


# 从CSV加载模型训练历史
def load_training_history(model_name):
    """
    从CSV文件加载模型的训练历史

    参数:
        model_name: 模型名称

    返回:
        包含训练历史的字典
    """
    csv_path = os.path.join(SAVE_DIR, f"{model_name}_training_history.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        history = {
            'epoch': df['epoch'].tolist(),
            'train_loss': df['train_loss'].tolist(),
            'train_acc': df['train_acc'].tolist(),
            'val_loss': df['val_loss'].tolist(),
            'val_acc': df['val_acc'].tolist()
        }
        return history
    else:
        print(f"找不到模型 {model_name} 的训练历史")
        return None


# 从CSV加载模型评估结果
def load_evaluation_results(model_name):
    """
    从CSV文件加载模型的评估结果

    参数:
        model_name: 模型名称

    返回:
        包含评估结果的字典
    """
    csv_path = os.path.join(SAVE_DIR, f"{model_name}_evaluation_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        results = df.to_dict('records')[0]
        return results
    else:
        print(f"找不到模型 {model_name} 的评估结果")
        return None


# 创建一个函数加载所有模型的训练历史和评估结果
def load_all_models_data(model_names):
    """
    加载所有模型的训练历史和评估结果

    参数:
        model_names: 模型名称列表

    返回:
        histories: 包含所有模型训练历史的列表
        eval_results: 包含所有模型评估结果的列表
    """
    histories = []
    eval_results = []

    for model_name in model_names:
        # 加载训练历史
        history = load_training_history(model_name)
        histories.append(history)

        # 加载评估结果
        results = load_evaluation_results(model_name)
        eval_results.append(results)

    return histories, eval_results


# 从保存的结果生成可视化
def generate_visualizations_from_saved_data(model_names):
    """
    从保存的数据生成所有可视化图表

    参数:
        model_names: 模型名称列表
    """
    # 加载所有模型数据
    histories, eval_results = load_all_models_data(model_names)

    # 确保所有数据都已加载
    if None in histories or None in eval_results:
        print("一些模型数据未找到，请确保先运行训练和评估")
        return

    # 提取准确率和类别准确率
    accuracies = [result['test_accuracy'] for result in eval_results]
    class_accuracies_list = []

    for result in eval_results:
        class_accs = []
        for class_name in CLASSES:
            class_accs.append(result[f'{class_name}_accuracy'])
        class_accuracies_list.append(class_accs)

    # 生成可视化
    print("生成训练历史对比图...")
    plot_training_history(histories, model_names)

    print("生成损失对比图...")
    plot_model_loss_comparison(histories, model_names)

    print("生成总体准确率对比图...")
    plot_model_overall_accuracy(model_names, accuracies)

    print("生成各类别准确率对比图...")
    plot_model_comparison_by_class(model_names, class_accuracies_list)

    print("可视化生成完成!")


# 汇总所有模型结果到一个CSV文件
def summarize_model_results(model_names):
    """
    汇总所有模型的评估结果到一个CSV文件

    参数:
        model_names: 模型名称列表
    """
    all_results = []

    for model_name in model_names:
        result = load_evaluation_results(model_name)
        if result:
            all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(SAVE_DIR, 'all_models_results.csv'), index=False)
        print(f"所有模型结果已汇总到 {os.path.join(SAVE_DIR, 'all_models_results.csv')}")
    else:
        print("没有找到任何模型结果")


def main():
    print("正在收集数据...")
    image_paths, labels = create_dataset_list(DATA_DIR, CLASSES)

    print("正在创建数据加载器...")
    batch_size = 256
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        image_paths, labels, batch_size=batch_size, test_size=0.1, val_size=0.1
    )

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 设置要训练的模型（只有VGG16使用预训练模型）
    models_to_train = {
        'VGG16': build_vgg16(NUM_CLASSES),
        'LeNet5': build_lenet5(NUM_CLASSES),
        'AlexNet': build_alexnet(NUM_CLASSES),
        'ResNet50': build_resnet50(NUM_CLASSES)
        # 'DenseNet121': build_densenet121(NUM_CLASSES)
    }

    # 设置训练参数
    num_epochs = 20
    criterion = LabelSmoothingLoss(classes=NUM_CLASSES, smoothing=0.1)

    # 存储训练历史和评估结果
    all_histories = []
    all_model_names = []
    all_test_accs = []
    all_class_accuracies = []
    all_confusion_matrices = []

    # 检查是否可以从保存的数据加载结果
    saved_data_exists = True
    for model_name in models_to_train.keys():
        if not os.path.exists(os.path.join(SAVE_DIR, f"{model_name}_evaluation_results.csv")):
            saved_data_exists = False
            break

    if saved_data_exists:
        print("发现已保存的模型评估结果，是否使用这些结果而不重新训练模型？(y/n)")
        response = input().strip().lower()

        if response == 'y':
            model_names = list(models_to_train.keys())
            print("从保存的数据生成可视化...")
            generate_visualizations_from_saved_data(model_names)
            print("汇总所有模型结果...")
            summarize_model_results(model_names)
            return

    # 训练和评估每个模型
    for model_name, model in models_to_train.items():
        print(f"\n开始训练模型: {model_name}")

        # 对于VGG16预训练模型使用较小的学习率
        if model_name == 'VGG16':
            feature_params = []
            classifier_params = []

            for name, param in model.named_parameters():
                if ('classifier' in name and '6' in name) or ('fc' in name):
                    classifier_params.append(param)
                else:
                    feature_params.append(param)

            optimizer = optim.Adam([
                {'params': feature_params, 'lr': 0.0001},
                {'params': classifier_params, 'lr': 0.001}
            ])
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # 训练模型
        trained_model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, model_name=model_name
        )

        # 评估模型
        print(f"\n评估模型: {model_name}")
        test_loss, test_acc, class_report, conf_matrix, class_accs = evaluate_model(
            trained_model, test_loader, criterion, model_name=model_name
        )

        # 存储结果
        all_histories.append(history)
        all_model_names.append(model_name)
        all_test_accs.append(test_acc)
        all_class_accuracies.append(class_accs)
        all_confusion_matrices.append(conf_matrix)

        # 保存模型到指定的路径
        save_path = os.path.join(SAVE_DIR, f"{model_name}_model.pth")
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'classes': CLASSES
        }, save_path)
        print(f"模型已保存至 {save_path}")

    # 绘制训练历史对比图
    print("\n绘制训练历史对比图...")
    plot_training_history(all_histories, all_model_names)

    # 绘制损失对比图
    print("\n绘制损失对比图...")
    plot_model_loss_comparison(all_histories, all_model_names)

    # 绘制总体准确率对比柱状图
    print("\n绘制总体准确率对比图...")
    plot_model_overall_accuracy(all_model_names, all_test_accs)

    # 绘制各类别准确率对比柱状图
    print("\n绘制各类别准确率对比图...")
    plot_model_comparison_by_class(all_model_names, all_class_accuracies)

    # 绘制每个模型的混淆矩阵
    print("\n绘制混淆矩阵...")
    for i, model_name in enumerate(all_model_names):
        plot_confusion_matrix(all_confusion_matrices[i], CLASSES, title=f"{model_name}_Confusion_Matrix")

    # 汇总所有模型结果
    print("\n汇总所有模型结果...")
    summarize_model_results(all_model_names)

    print("\n所有模型训练和评估完成!")


if __name__ == "__main__":
    main()
