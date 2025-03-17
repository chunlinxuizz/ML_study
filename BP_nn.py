import numpy as np
import random
import math

# 生成 [a, b] 范围内的随机数
def rand(a, b):
    return (b - a) * random.random() + a

# 创建一个 m x n 的矩阵，并用 fill 填充
def make_matrix(m, n, fill=0.0):
    return [[fill] * n for _ in range(m)]

# Sigmoid 激活函数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# Sigmoid 函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)

# 反向传播神经网络类
class BPNeuralNetwork:
    def __init__(self):
        # 初始化网络参数
        self.input_n = 0        # 输入层节点数（包括偏置节点）
        self.hidden_n = 0       # 隐藏层节点数
        self.output_n = 0       # 输出层节点数
        self.input_cells = []   # 输入层节点的值
        self.hidden_cells = []  # 隐藏层节点的值
        self.output_cells = []  # 输出层节点的值
        self.input_weights = [] # 输入层到隐藏层的权重矩阵
        self.output_weights = [] # 隐藏层到输出层的权重矩阵

    # 设置网络结构并初始化权重
    def setup(self, ni, nh, no):
        self.input_n = ni + 1   # 输入层节点数（包括偏置节点）
        self.hidden_n = nh      # 隐藏层节点数
        self.output_n = no      # 输出层节点数
        self.input_cells = [1.0] * self.input_n  # 初始化输入层节点值
        self.hidden_cells = [1.0] * self.hidden_n  # 初始化隐藏层节点值
        self.output_cells = [1.0] * self.output_n  # 初始化输出层节点值
        self.input_weights = make_matrix(self.input_n, self.hidden_n)  # 初始化输入层到隐藏层的权重矩阵
        self.output_weights = make_matrix(self.hidden_n, self.output_n)  # 初始化隐藏层到输出层的权重矩阵

        # 初始化输入层到隐藏层的权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)  # 随机初始化权重

        # 初始化隐藏层到输出层的权重
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)  # 随机初始化权重

    # 前向传播，计算输出
    def predict(self, inputs):
        # 设置输入层节点的值（不包括偏置节点）
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        # 计算隐藏层节点的值
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]  # 加权求和
            self.hidden_cells[j] = sigmoid(total)  # 使用 Sigmoid 激活函数

        # 计算输出层节点的值
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]  # 加权求和
            self.output_cells[k] = sigmoid(total)  # 使用 Sigmoid 激活函数

        # 返回输出层节点的值
        return self.output_cells[:]

    # 反向传播，更新权重
    def back_propagate(self, case, label, learn):
        # 前向传播，计算输出
        self.predict(case)

        # 初始化输出层的误差
        output_deltas = [0.0] * self.output_n

        # 计算输出层的误差
        for k in range(self.output_n):
            error = label[k] - self.output_cells[k]  # 目标输出与实际输出的差值
            output_deltas[k] = sigmoid_derivative(self.output_cells[k]) * error  # 误差乘以 Sigmoid 导数

        # 初始化隐藏层的误差
        hidden_deltas = [0.0] * self.hidden_n

        # 计算隐藏层的误差
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.output_n):
                error += output_deltas[k] * self.output_weights[j][k]  # 将输出层的误差传播到隐藏层
            hidden_deltas[j] = sigmoid_derivative(self.hidden_cells[j]) * error  # 误差乘以 Sigmoid 导数

        # 更新隐藏层到输出层的权重
        for j in range(self.hidden_n):
            for k in range(self.output_n):
                self.output_weights[j][k] += learn * output_deltas[k] * self.hidden_cells[j]  # 权重更新

        # 更新输入层到隐藏层的权重
        for i in range(self.input_n):
            for j in range(self.hidden_n):
                self.input_weights[i][j] += learn * hidden_deltas[j] * self.input_cells[i]  # 权重更新

        # 计算总误差
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2  # 均方误差

        # 返回总误差
        return error

    # 训练网络
    def train(self, cases, labels, limit=10000, learn=0.1):
        for i in range(limit):
            error = 0
            for j in range(len(cases)):
                label = labels[j]
                case = cases[j]
                error += self.back_propagate(case, label, learn)  # 反向传播并累加误差
            if i % 1000 == 0:
                print(f"Iteration {i}, Error: {error}")  # 每 1000 次迭代打印一次误差

    # 测试网络
    def test(self):
        # 定义测试数据（XOR 问题）
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]  # XOR 问题的目标输出

        # 设置网络结构（2 输入，10 隐藏层节点，1 输出）
        self.setup(2, 10, 1)

        # 训练网络
        self.train(cases, labels, 10000, 0.1)

        # 测试网络
        for case in cases:
            print(self.predict(case))  # 打印每个测试用例的输出

# 主程序
if __name__ == '__main__':
    nn = BPNeuralNetwork()  # 创建神经网络对象
    nn.test()  # 测试网络
