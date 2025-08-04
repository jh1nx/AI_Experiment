import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures

# 设置matplotlib绘图中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   

# 读取数据函数
def load_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            data.append([float(x) for x in row])  # 转换为浮点数
    return np.array(data)

# 异常值检测与处理
def detect_and_remove_outliers(X, y, method='iqr', threshold=1.5):
    data = np.hstack((X, y.reshape(-1, 1)))
    n_samples, n_features = data.shape
    outliers_mask = np.zeros(n_samples, dtype=bool)
    
    if method == 'iqr':
        for i in range(n_features):
            q1 = np.percentile(data[:, i], 25)  
            q3 = np.percentile(data[:, i], 75)  
            iqr = q3 - q1 
            lower_bound = q1 - threshold * iqr 
            upper_bound = q3 + threshold * iqr 
            column_outliers = (data[:, i] < lower_bound) | (data[:, i] > upper_bound)  
            outliers_mask = outliers_mask | column_outliers  
    elif method == 'zscore':
        for i in range(n_features):
            z_scores = np.abs(stats.zscore(data[:, i]))  
            column_outliers = z_scores > threshold 
            outliers_mask = outliers_mask | column_outliers 
    
    return X[~outliers_mask], y[~outliers_mask], outliers_mask

# 数据标准化
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def transform_geo_features(X, y=None, cluster_centers=None, n_clusters=5):
    geo_coords = X[:, :2].copy()
    
    if cluster_centers is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(geo_coords)
        cluster_centers = kmeans.cluster_centers_
    else:
        kmeans = KMeans(n_clusters=len(cluster_centers), random_state=42)
        kmeans.cluster_centers_ = cluster_centers
    
    dist_to_clusters = np.zeros((len(X), n_clusters))
    for i in range(n_clusters):
        center = cluster_centers[i]
        dist_to_clusters[:, i] = np.sqrt(np.sum((geo_coords - center) ** 2, axis=1))
    
    other_features = X[:, 2:].copy()
    X_transformed = np.hstack((other_features, dist_to_clusters))
    
    return X_transformed, cluster_centers

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.001, epochs=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        for i in range(1, self.num_layers):
            w = np.random.randn(self.layer_sizes[i-1], self.layer_sizes[i]) * np.sqrt(2.0 / self.layer_sizes[i-1])
            b = np.zeros((1, self.layer_sizes[i]))
            self.weights.append(w)
            self.biases.append(b)
    
    def tanh(self, x):
        """双曲正切激活函数"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """双曲正切激活函数的导数"""
        return 1.0 - np.tanh(x)**2
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        return np.where(x > 0, 1.0, 0.0)
    
    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU激活函数"""
        return np.maximum(alpha * x, x)
    
    def leaky_relu_derivative(self, x, alpha=0.01):
        """Leaky ReLU激活函数的导数"""
        return np.where(x > 0, 1.0, alpha)
    
    def linear(self, x):
        """线性激活函数，直接返回输入"""
        return x
    
    def linear_derivative(self, x):
        """线性激活函数的导数"""
        return np.ones_like(x)
    
    def forward(self, X):
        activations = [X]
        layer_inputs = []
        
        for i in range(self.num_layers - 1):
            layer_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(layer_input)
            
            if i == 0:
                activation = self.tanh(layer_input)
            elif i == self.num_layers - 2:
                activation = self.linear(layer_input)
            elif i == 1:
                activation = self.relu(layer_input)
            elif i == 2 and self.num_layers > 4:
                activation = self.leaky_relu(layer_input)
            elif i == 3 and self.num_layers > 5:
                activation = self.tanh(layer_input)
            else:
                activation = self.relu(layer_input)
            activations.append(activation)
            
        return activations, layer_inputs
    
    def backward(self, X, y, activations, layer_inputs):
        """
        反向传播，计算梯度并更新权重
        参数:
        X: 输入特征矩阵
        y: 目标向量
        activations: 前向传播中每层的激活值
        layer_inputs: 前向传播中每层的输入值
        """
        n_samples = X.shape[0]
        
        # 计算输出层误差 (y_pred - y_true)
        output_error = activations[-1] - y.reshape(-1, 1)
        
        # 初始化当前误差为输出层误差
        delta = output_error
        
        # 从输出层向前反向传播
        for i in range(self.num_layers - 2, -1, -1):
            # 计算当前层权重的梯度
            dw = np.dot(activations[i].T, delta) / n_samples
            db = np.sum(delta, axis=0, keepdims=True) / n_samples
            
            # 更新当前层的权重和偏置
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            # 如果不是第一层，则计算前一层的误差
            if i > 0:
                # 确定当前层使用的激活函数的导数
                if i == 1:  # 第一隐藏层使用tanh
                    derivative = self.tanh_derivative(layer_inputs[i-1])
                elif i == 2 and self.num_layers > 4:  # 第三隐藏层使用Leaky ReLU
                    derivative = self.leaky_relu_derivative(layer_inputs[i-1])
                elif i == 3 and self.num_layers > 5:  # 第四隐藏层(新增)使用tanh
                    derivative = self.tanh_derivative(layer_inputs[i-1])
                else:  # 其他隐藏层使用ReLU
                    derivative = self.relu_derivative(layer_inputs[i-1])
                
                # 计算前一层的误差
                delta = np.dot(delta, self.weights[i].T) * derivative

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # 前向传播
            activations, layer_inputs = self.forward(X)
            
            # 计算损失(均方误差)
            predictions = activations[-1]
            loss = np.mean((predictions - y.reshape(-1, 1)) ** 2)
            self.loss_history.append(loss)
            
            # 反向传播
            self.backward(X, y, activations, layer_inputs)
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1].flatten()  # 返回一维数组形式的输出

print("加载数据...")
data = load_data('MLP_data.csv')  # 从CSV文件加载数据集

# 划分特征和目标
X = data[:, :4]  # 特征：longitude, latitude, housing_age, homeowner_income
y = data[:, 4]   # 目标变量：house_price

print("检测并处理异常值...")
original_data_size = X.shape[0]  # 原始样本数量

# 应用IQR方法处理异常值 - threshold=1.5是箱线图异常值检测的标准阈值
X_clean, y_clean, outliers_mask = detect_and_remove_outliers(X, y, method='iqr', threshold=1.5)
cleaned_data_size = X_clean.shape[0]  # 处理后的样本数量
outliers_removed = original_data_size - cleaned_data_size  # 移除的异常样本数量
print(f"原始数据样本数: {original_data_size}")
print(f"处理后数据样本数: {cleaned_data_size}")
print(f"被移除的异常样本数: {outliers_removed} ({outliers_removed/original_data_size*100:.2f}%)")

# 清理前后对比
plt.figure(figsize=(20, 15))
plt.suptitle('数据清理前后对比', fontsize=20)

feature_names = ['经度', '纬度', '房龄', '收入']
for i in range(4):
    # 原始数据散点图
    plt.subplot(2, 4, i+1) 
    plt.scatter(X[:, i], y, alpha=0.5, c='blue', label='所有数据点') 
    plt.scatter(X[outliers_mask, i], y[outliers_mask], alpha=0.7, c='red', label='异常数据点')  # 突出显示异常点
    plt.title(f'清理前: {feature_names[i]}与房价关系', fontsize=12)
    plt.xlabel(feature_names[i])
    plt.ylabel('房价')
    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend()
    
    # 清理后的数据散点图
    plt.subplot(2, 4, i+5) 
    plt.scatter(X_clean[:, i], y_clean, alpha=0.5, c='green', label='保留的数据点') 
    plt.title(f'清理后: {feature_names[i]}与房价关系', fontsize=12)
    plt.xlabel(feature_names[i])
    plt.ylabel('房价')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # 添加图例

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig('data_cleaning_comparison.png')
plt.show()

# 转换地理特征 - 处理经度和纬度的非线性关系
X_transformed, cluster_centers = transform_geo_features(X_clean, y_clean, n_clusters=5)


# 可视化聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean, cmap='viridis', alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=100)

# 在聚类中心附近标注区域编号
for i, center in enumerate(cluster_centers):
    plt.annotate(f'区域{i+1}', xy=(center[0], center[1]), 
                 xytext=(center[0]+0.02, center[1]+0.02),
                 color='red', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.colorbar(label='房价')
plt.title('基于地理位置的聚类分析')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('geo_clustering.png')
plt.show()

# 标准化数据 - 使得每个特征均值为0，标准差为1
print("数据标准化...")
X_norm, X_mean, X_std = normalize_data(X_transformed)

# 标准化目标变量y
y_norm, y_mean, y_std = normalize_data(y_clean.reshape(-1, 1))
y_norm = y_norm.flatten()  # 将二维数组转回一维

# 划分(80% 训练, 20% 测试)
np.random.seed(42)  # 设置随机种子，确保结果可复现
indices = np.random.permutation(len(X_norm))  # 生成随机排列的索引
train_size = int(0.8 * len(X_norm)) 
train_indices = indices[:train_size] 
test_indices = indices[train_size:] 
X_train, y_train = X_norm[train_indices], y_norm[train_indices]
X_test, y_test = X_norm[test_indices], y_norm[test_indices]

# 定义两层隐藏层的网络架构和参数组合
learning_rates = [0.05,0.02,0.01 ] # 学习率
epochs_list = [1000,2000,5000]  # 迭代次数
# 只保留两层隐藏层的网络
two_layer_architecture = [X_train.shape[1], 20, 10, 1]  # 输入层-20神经元隐藏层-10神经元隐藏层-输出层

models = []  # 存储训练好的模型
results = []  # 存储各模型的评估结果

# 创建一个图用于显示损失曲线
plt.figure(figsize=(15, 8))

# 计数器用于子图位置
plot_idx = 1

# 指定的最佳参数组合
best_model_index = -1
best_lr = 0.1
best_epochs = 5000

# 训练两层隐藏层MLP模型，使用不同的学习率和迭代次数
print(f"训练两层隐藏层MLP模型 (架构: {'-'.join(map(str, two_layer_architecture))})")
arch_name = f"2层隐藏层 ({'-'.join(map(str, two_layer_architecture))})"

for lr_idx, lr in enumerate(learning_rates):
    for epochs_idx, epochs in enumerate(epochs_list):
        print(f"  训练模型 - 学习率: {lr}, 迭代次数: {epochs}")
        
        # 训练模型
        model = MLP(layer_sizes=two_layer_architecture, learning_rate=lr, epochs=epochs)
        model.fit(X_train, y_train)
        models.append(model)
        
        # 评估模型
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        
        # 计算原始尺度的MSE（反标准化后）
        y_pred_orig = y_pred * y_std + y_mean
        y_test_orig = y_test * y_std + y_mean
        orig_mse = np.mean((y_pred_orig - y_test_orig) ** 2)
        
        # 检查是否为指定的最佳参数组合
        if lr == best_lr and epochs == best_epochs:
            best_model_index = len(models) - 1
        
        results.append({
            'learning_rate': lr,
            'epochs': epochs,
            'mse': mse,
            'orig_mse': orig_mse,
            'index': len(models) - 1
        })
        
        # 绘制损失曲线子图
        if plot_idx <= 9:  # 控制最多9个子图
            plt.subplot(3, 3, plot_idx)
            plt.plot(model.loss_history)
            plt.title(f'学习率={lr}, 迭代={epochs}')
            plt.xlabel('迭代次数')
            plt.ylabel('MSE')
            plt.grid(True)
            plot_idx += 1

plt.suptitle('两层隐藏层MLP模型不同参数的损失曲线', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('two_layer_mlp_parameters_loss.png')
plt.show()

# 选择最佳模型
if best_model_index >= 0:
    best_result = results[best_model_index]
    best_model = models[best_model_index]
else:
    # 如果没有匹配到指定的参数组合，选择MSE最小的模型
    best_model_index = np.argmin([result['mse'] for result in results])
    best_result = results[best_model_index]
    best_model = models[best_model_index]

# 输出不同参数组合的结果
print("\n两层隐藏层MLP模型不同参数组合的结果:")
for result in results:
    print(f"学习率: {result['learning_rate']}, 迭代次数: {result['epochs']}, "
          f"标准化MSE: {result['mse']:.6f}, 原始MSE: {result['orig_mse']:.2f}")
print(f"\n选定的最佳两层隐藏层MLP模型参数 - 学习率: {best_result['learning_rate']}, "
      f"迭代次数: {best_result['epochs']}, 标准化MSE: {best_result['mse']:.6f}")

# 计算选定最佳模型的R²
y_pred_best = best_model.predict(X_test)
y_pred_best_orig = y_pred_best * y_std + y_mean  # 反标准化
ss_total = np.sum((y_test_orig - np.mean(y_test_orig)) ** 2)  # 总平方和
ss_residual = np.sum((y_test_orig - y_pred_best_orig) ** 2)  # 残差平方和
r_squared = 1 - (ss_residual / ss_total)  # R² = 1 - SSR/SST
print(f"最佳两层隐藏层MLP模型 R²(决定系数): {r_squared:.4f}")

# 绘制不同参数组合的MSE条形图
plt.figure(figsize=(12, 6))
x = np.arange(len(results))
plt.bar(x, [result['mse'] for result in results])
xlabels = [f"lr={result['learning_rate']}, ep={result['epochs']}" for result in results]
plt.xticks(x, xlabels, rotation=45, ha='right')
plt.title('两层隐藏层MLP模型不同参数组合的标准化均方误差比较')
plt.ylabel('标准化MSE')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('two_layer_mlp_mse_comparison.png')
plt.show()

# 可视化最佳两层隐藏层MLP模型的预测结果
plt.figure(figsize=(12, 10))
# 1. 真实房价与预测房价的散点图比较
plt.subplot(2, 2, 1)
plt.scatter(y_test_orig, y_pred_best_orig, alpha=0.5)
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], 'r--')
plt.title(f'真实房价 vs 预测房价\n(学习率={best_result["learning_rate"]}, 迭代={best_result["epochs"]})')
plt.xlabel('真实房价')
plt.ylabel('预测房价')

# 2. 收入与房价关系的散点图
plt.subplot(2, 2, 2)
# 获取原始尺度的收入数据（未标准化）
income_orig = X_clean[test_indices, 3] * X_std[3] + X_mean[3] if len(X_std) > 3 else X_clean[test_indices, 3]
plt.scatter(income_orig, y_test_orig, alpha=0.5, label='真实值', c='blue')
plt.scatter(income_orig, y_pred_best_orig, alpha=0.5, label='预测值', c='orange')
plt.title('收入与房价关系')
plt.xlabel('收入（标准化前）')
plt.ylabel('房价')
plt.legend()

# 3. 房龄与房价关系的散点图
plt.subplot(2, 2, 3)
# 获取原始尺度的房龄数据（未标准化）
age_orig = X_clean[test_indices, 2] * X_std[2] + X_mean[2] if len(X_std) > 2 else X_clean[test_indices, 2]
plt.scatter(age_orig, y_test_orig, alpha=0.5, label='真实值', c='blue')
plt.scatter(age_orig, y_pred_best_orig, alpha=0.5, label='预测值', c='orange')
plt.title('房龄与房价关系')
plt.xlabel('房龄（标准化前）')
plt.ylabel('房价')
plt.legend()

# 4. 地理位置(经度、纬度)与预测房价关系的热力散点图
plt.subplot(2, 2, 4)
sc = plt.scatter(X_clean[test_indices, 0], X_clean[test_indices, 1], c=y_pred_best_orig, cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='预测房价')
plt.title('地理位置与预测房价关系')
plt.xlabel('经度')
plt.ylabel('纬度')

# 调整布局并保存图片
plt.tight_layout()
plt.savefig('best_model_predictions.png')
plt.show()

def visualize_network_architecture(model, title='神经网络架构'):
    plt.figure(figsize=(10, 6))
    layer_sizes = model.layer_sizes
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    layer_names = ['输入层'] 
    for i in range(1, n_layers-1):
        if i == 1:
            layer_names.append('隐藏层1\n(tanh激活)')
        elif i == n_layers-2:
            layer_names.append(f'隐藏层{i}\n(ReLU激活)')
        else:
            layer_names.append(f'隐藏层{i}\n(ReLU激活)')
    layer_names.append('输出层\n(线性激活)')
    
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        x = i/(n_layers-1)
        neurons_y = np.linspace(0, 1, size+2)[1:-1]
        
        for j, y in enumerate(neurons_y):
            circle = plt.Circle((x, y), 0.02, color='blue', fill=True)
            plt.gca().add_patch(circle)
            
            if i < n_layers-1:
                next_size = layer_sizes[i+1]
                next_neurons_y = np.linspace(0, 1, next_size+2)[1:-1]
                next_x = (i+1)/(n_layers-1)
                for k, next_y in enumerate(next_neurons_y):
                    plt.plot([x, next_x], [y, next_y], 'gray', alpha=0.3)
        
        plt.text(x, -0.05, name, ha='center')
        plt.text(x, 1.05, f'{size}个神经元', ha='center')
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('network_architecture.png')
    plt.show()

# 主程序
if __name__ == "__main__":
    print("加载数据...")
    data = load_data('MLP_data.csv')
    
    # 修改后的可视化调用
    arch_str = '-'.join(map(str, best_model.layer_sizes))
    visualize_network_architecture(best_model, title=f'最佳模型架构 - {arch_str}')
    
    print("实现两层隐藏层MLP，具有以下激活函数策略:")
    print("• 第一隐藏层：tanh激活函数")
    print("• 第二隐藏层：ReLU激活函数")
    print("• 输出层：线性激活函数")