import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from time import time
import pandas as pd

# ================== 超参数配置 ==================
class Config:
    # 数据参数
    class_names = ["baihe", "dangshen", "gouqi", "huaihua", "jinyinhua"]
    num_classes = len(class_names)
    
    # 训练参数
    batch_size = 64  # 平衡内存和性能
    lr = 0.001
    epochs = 20
    step_size = 5     # 学习率衰减步长
    gamma = 0.1       # 学习率衰减系数
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置（建议通过构造函数设置）
    def __init__(self, data_dir):
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.results_dir = os.path.join(data_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

# ================== 数据加载 ==================
class SmartDataset(Dataset):
    """智能数据集类：支持文件夹和文件名两种组织方式"""
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir: 数据根目录
            transform: 数据预处理
            mode: 'train'（按文件夹分类）或 'test'（按文件名分类）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        if self.mode == 'train':
            # 训练模式：按文件夹分类
            for class_idx, class_name in enumerate(Config.class_names):
                class_dir = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            samples.append((
                                os.path.join(class_dir, img_name),
                                class_idx
                            ))
        else:
            # 测试模式：按文件名分类
            for img_name in os.listdir(self.root_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(self.root_dir, img_name)
                    for class_idx, class_name in enumerate(Config.class_names):
                        if class_name.lower() in img_name.lower():
                            samples.append((img_path, class_idx))
                            break
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)   
        return img, label

# ================== 模型架构 ==================
class EnhancedCNN(nn.Module):
    """增强版CNN：结合深度特征提取与高效分类器"""
    def __init__(self, num_classes):
        super().__init__()
        # 特征提取器（5层卷积）
        self.features = nn.Sequential(
            # 卷积块1 (224x224 -> 112x112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块2 (112x112 -> 56x56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块3 (56x56 -> 28x28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块4 (28x28 -> 14x14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 卷积块5 (14x14 -> 7x7)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 自适应池化 (7x7 -> 1x1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器（带Dropout）
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 适度正则化
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================== 训练工具 ==================
class Trainer:
    """集成化训练器：封装训练逻辑和可视化"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': []
        }
        
        # 批次级别的损失跟踪
        self.batch_losses = []
        self.best_epoch_batch_losses = None
        self.best_epoch = -1
        
        # 初始化优化器和学习率调度
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = StepLR(self.optimizer, 
                              step_size=config.step_size, 
                              gamma=config.gamma)
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        # 数据加载
        self._init_dataloaders()
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        train_set = SmartDataset(
            self.config.train_dir, 
            transform=self.transform,
            mode='train'
        )
        test_set = SmartDataset(
            self.config.test_dir,
            transform=self.transform,
            mode='test'
        )
        
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2  # 减少工作线程数以避免潜在问题
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"训练集: {len(train_set)} 图像, 测试集: {len(test_set)} 图像")
    
    def train_epoch(self, epoch):
        """执行一个epoch的训练，并记录批次级别的损失"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_batch_losses = []  # 记录当前epoch的每个batch的loss
        
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录当前batch的loss
            batch_loss = loss.item()
            epoch_batch_losses.append(batch_loss)
            
            # 为整体批次损失列表添加当前batch loss和epoch信息
            self.batch_losses.append({
                'epoch': epoch + 1,
                'batch': batch_idx + 1,
                'loss': batch_loss
            })
            
            # 统计指标
            running_loss += batch_loss
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算epoch指标
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, epoch_batch_losses
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss = running_loss / len(self.test_loader)
        test_acc = correct / total
        return test_loss, test_acc
    
    def run(self):
        """执行完整训练流程"""
        best_acc = 0.0
        start_time = time()
        
        print(f"训练设备: {self.config.device}")
        print(f"学习率: {self.config.lr}")
        print(f"开始训练 {self.config.epochs} 个周期...")
        
        # 用于记录详细训练信息的表格数据
        detailed_history = []
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time()
            
            # 训练和评估
            print(f"\nEpoch {epoch+1}/{self.config.epochs}:")
            train_loss, train_acc, epoch_batch_losses = self.train_epoch(epoch)
            test_loss, test_acc = self.evaluate()
            
            epoch_duration = time() - epoch_start_time
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            # 记录详细训练信息
            detailed_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': current_lr,
                'time_sec': epoch_duration
            })
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                self.best_epoch = epoch
                self.best_epoch_batch_losses = epoch_batch_losses  # 保存最佳epoch的batch losses
                torch.save(self.model.state_dict(), 
                         os.path.join(self.config.results_dir, 'best_model.pth'))
            
            # 打印进度
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"时间: {epoch_duration:.1f}s")
        
        # 训练结束处理
        duration = time() - start_time
        print(f"\n训练完成，总用时 {duration:.2f} 秒")
        print(f"最佳测试准确率: {best_acc:.4f} (Epoch {self.best_epoch+1})")
        
        # 保存训练历史到CSV
        self.save_training_history(detailed_history)
        
        # 可视化结果
        print("正在生成可视化结果...")
        self.visualize_results(duration, detailed_history)
        print(f"可视化结果已保存到: {self.config.results_dir}")
        
    def save_training_history(self, detailed_history):
        """保存训练历史记录到CSV文件"""
        df = pd.DataFrame(detailed_history)
        csv_path = os.path.join(self.config.results_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        
    def plot_batch_losses(self):
        """绘制批次级别的损失曲线"""
        try:
            # 提取批次损失数据
            df = pd.DataFrame(self.batch_losses)
            
            plt.figure(figsize=(10, 5))
            
            # 将批次索引转换为全局批次计数
            df['global_batch'] = df.index
            
            # 绘制所有批次的损失曲线
            sns.lineplot(x='global_batch', y='loss', data=df, alpha=0.7)
            
            plt.title('Batch Losses During Training')
            plt.xlabel('Batch Count')
            plt.ylabel('Loss Value')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'batch_losses.png'))
            plt.close()
        except Exception as e:
            print(f"绘制批次损失图时出错: {e}")
    
    def create_metrics_table(self, detailed_history):
        """创建训练指标表格"""
        try:
            # 将训练历史数据转换为DataFrame
            df = pd.DataFrame(detailed_history)
            
            # 格式化数据
            df_display = df.copy()
            df_display['train_loss'] = df_display['train_loss'].map('{:.4f}'.format)
            df_display['train_acc'] = df_display['train_acc'].map('{:.4f}'.format)
            df_display['test_loss'] = df_display['test_loss'].map('{:.4f}'.format)
            df_display['test_acc'] = df_display['test_acc'].map('{:.4f}'.format)
            df_display['lr'] = df_display['lr'].map('{:.6f}'.format)
            df_display['time_sec'] = df_display['time_sec'].map('{:.1f}'.format)
            
            # 生成表格
            fig, ax = plt.subplots(figsize=(10, len(detailed_history) * 0.4 + 1.5))
            ax.axis('off')
            
            # 创建表格
            table = ax.table(
                cellText=df_display.values,
                colLabels=df_display.columns,
                loc='center',
                cellLoc='center'
            )
            
            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.2)
            
            # 高亮最佳epoch行
            best_epoch_idx = self.best_epoch
            if 0 <= best_epoch_idx < len(detailed_history):
                for j in range(len(df.columns)):
                    table[(best_epoch_idx + 1, j)].set_facecolor('#90EE90')
            
            plt.title('Training Metrics Details')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'metrics_table.png'), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"创建指标表格时出错: {e}")

    def visualize_results(self, duration, detailed_history):
        """简化版可视化训练结果"""
        try:
            # 1. 训练曲线
            plt.figure(figsize=(12, 5))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(self.history['train_loss'], label='Train')
            plt.plot(self.history['test_loss'], label='Test')
            plt.title('Loss Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 准确率曲线
            plt.subplot(1, 2, 2)
            plt.plot(self.history['train_acc'], label='Train')
            plt.plot(self.history['test_acc'], label='Test')
            plt.title('Accuracy Curves')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.results_dir, 'training_curves.png'))
            plt.close()
        
            # 2. 生成批次级别的损失曲线
            self.plot_batch_losses()

            # 3. 保存批次级别损失图
            self.save_training_history(detailed_history)
            
            # 4. 训练指标表格
            self.create_metrics_table(detailed_history)
            
            # 5. 混淆矩阵
            self.plot_confusion_matrix()
        except Exception as e:
            print(f"可视化结果时出错: {e}")
     
    def compute_confusion_matrix(self):
        """计算混淆矩阵"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return confusion_matrix(all_labels, all_preds)
    
    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        try:
            # 计算混淆矩阵
            cm = self.compute_confusion_matrix()
            
            # 创建图形
            plt.figure(figsize=(10, 8))
            
            # 使用seaborn绘制更美观的混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.config.class_names,
                        yticklabels=self.config.class_names)
            
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # 保存图形
            plt.savefig(os.path.join(self.config.results_dir, 'confusion_matrix.png'))
            plt.close()
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {e}")

# ================== 主程序 ==================
if __name__ == "__main__":
    try:
        # 初始化配置（修改为你的数据目录）
        config = Config(data_dir="E:/大学课件/人工智能作业/7 深度学习/mine")
        
        # 初始化模型
        model = EnhancedCNN(config.num_classes).to(config.device)
        
        # 创建训练器并运行
        trainer = Trainer(model, config)
        trainer.run()
    except Exception as e:
        print(f"程序运行出错: {e}")