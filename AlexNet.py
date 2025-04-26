import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from common import SoftmaxWithLoss, Relu, Adam, Affine, Convolution, Pooling, Dropout


class AlexNet:
    """AlexNet实现

    网络结构：
    卷积1 - ReLU - 池化 - 卷积2 - ReLU - 池化 -
    卷积3 - ReLU - 卷积4 - ReLU - 卷积5 - ReLU - 池化 -
    全连接1 - ReLU - Dropout - 全连接2 - ReLU - Dropout - 全连接3 - softmax

    参数说明
    ----------
    input_dim : 输入维度 (MNIST为(1, 28, 28))
    output_size : 输出类别数 (MNIST为10))
    weight_init_std : 权重初始化标准差
    dropout_ratio : Dropout比例
    """

    def __init__(self, input_dim=(1, 28, 28), output_size=10,
                 weight_init_std=0.01, dropout_ratio=0.5):
        # 初始化权重参数
        self.params = {}

        # 第一卷积层：32个3x3滤波器 (原AlexNet是96个11x11)
        self.params['W1'] = weight_init_std * np.random.randn(32, input_dim[0], 3, 3)
        self.params['b1'] = np.zeros(32)

        # 第二卷积层：64个3x3滤波器
        self.params['W2'] = weight_init_std * np.random.randn(64, 32, 3, 3)
        self.params['b2'] = np.zeros(64)

        # 第三卷积层：128个3x3滤波器
        self.params['W3'] = weight_init_std * np.random.randn(128, 64, 3, 3)
        self.params['b3'] = np.zeros(128)

        # 第四卷积层：128个3x3滤波器
        self.params['W4'] = weight_init_std * np.random.randn(128, 128, 3, 3)
        self.params['b4'] = np.zeros(128)

        # 第五卷积层：64个3x3滤波器
        self.params['W5'] = weight_init_std * np.random.randn(64, 128, 3, 3)
        self.params['b5'] = np.zeros(64)

        # 第一全连接层：1024个神经元 (原AlexNet是4096)
        self.params['W6'] = weight_init_std * np.random.randn(64 * 3 * 3, 1024)
        self.params['b6'] = np.zeros(1024)

        # 第二全连接层：512个神经元
        self.params['W7'] = weight_init_std * np.random.randn(1024, 512)
        self.params['b7'] = np.zeros(512)

        # 输出层
        self.params['W8'] = weight_init_std * np.random.randn(512, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 构建网络层
        self.layers = OrderedDict()

        # 第一卷积块
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           stride=1, pad=1)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第二卷积块
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           stride=1, pad=1)
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第三卷积层
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           stride=1, pad=1)
        self.layers['Relu3'] = Relu()

        # 第四卷积层
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'],
                                           stride=1, pad=1)
        self.layers['Relu4'] = Relu()

        # 第五卷积块
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'],
                                           stride=1, pad=1)
        self.layers['Relu5'] = Relu()
        self.layers['Pool5'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # 第一全连接块
        self.layers['Affine1'] = Affine(self.params['W6'], self.params['b6'])
        self.layers['Relu6'] = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio)

        # 第二全连接块
        self.layers['Affine2'] = Affine(self.params['W7'], self.params['b7'])
        self.layers['Relu7'] = Relu()
        self.layers['Dropout2'] = Dropout(dropout_ratio)

        # 输出层
        self.layers['Affine3'] = Affine(self.params['W8'], self.params['b8'])

        self.last_layer = SoftmaxWithLoss()

        self.use_dropout = True if dropout_ratio > 0.0 else False

    def predict(self, x, train_flg=False):
        """前向传播"""
        for key, layer in self.layers.items():
            if 'Dropout' in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        """计算损失函数"""
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """计算准确率"""
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """使用反向传播计算梯度"""
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 保存梯度
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Conv2'].dW
        grads['b2'] = self.layers['Conv2'].db
        grads['W3'] = self.layers['Conv3'].dW
        grads['b3'] = self.layers['Conv3'].db
        grads['W4'] = self.layers['Conv4'].dW
        grads['b4'] = self.layers['Conv4'].db
        grads['W5'] = self.layers['Conv5'].dW
        grads['b5'] = self.layers['Conv5'].db
        grads['W6'] = self.layers['Affine1'].dW
        grads['b6'] = self.layers['Affine1'].db
        grads['W7'] = self.layers['Affine2'].dW
        grads['b7'] = self.layers['Affine2'].db
        grads['W8'] = self.layers['Affine3'].dW
        grads['b8'] = self.layers['Affine3'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        """保存模型参数"""
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """加载模型参数"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5',
                                 'Affine1', 'Affine2', 'Affine3']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# 加载MNIST数据集
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# 创建AlexNet实例
network = AlexNet(input_dim=(1, 28, 28), output_size=10, weight_init_std=0.01, dropout_ratio=0.5)

# 训练参数设置
epochs = 10  # 训练轮数
train_size = x_train.shape[0]  # 训练数据大小
batch_size = 32  # 批大小
learning_rate = 0.001  # 学习率(比原SimpleConvNet更小)

# 记录训练过程
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算每个epoch的迭代次数
iter_per_epoch = max(train_size / batch_size, 1)
iters_num = int(epochs * iter_per_epoch + 1)

# 训练循环
for i in range(iters_num):
    # 随机选择批量数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 使用Adam优化器更新参数
    optim = Adam()
    optim.update(network.params, grad)

    # 记录损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每个epoch计算并打印准确率和损失值
    if i % iter_per_epoch == 0:
        epoch_loss = np.mean(train_loss_list[-int(iter_per_epoch):])  # 直接从 train_loss_list 中提取当前 epoch 的平均损失
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"epoch: {i // iter_per_epoch}, 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}, 平均损失: {epoch_loss:.6f}")


# 绘制训练准确率和测试准确率曲线
plt.figure(figsize=(8, 5))  # 创建第一个图形
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train-acc', color='blue')
plt.plot(x, test_acc_list, label='test-acc', linestyle='--', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('B20220307114-AlexNet')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig('AlexNet_Accuracy.png', dpi=300)
plt.show()  # 显示第一个弹窗

# 绘制训练损失曲线
plt.figure(figsize=(8, 5))  # 创建第二个图形
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='Train Loss', color='green')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title('Training Loss')
plt.legend(loc='upper right')
plt.savefig('AlexNet_Loss.png', dpi=300)
plt.show()  # 显示第二个弹窗