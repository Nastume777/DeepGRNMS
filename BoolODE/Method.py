import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

def generate_expression_data(filename):
    expression_data = pd.read_csv(filename, header='infer', index_col=0)
    expression_data = np.array(expression_data, dtype=np.float64)
    return expression_data

def generate_pair_data(data):
    data_tf = data.copy()
    maxnum = np.ceil(np.max(np.max(data_tf)))
    num = input_size / maxnum
    data_tf = np.around(data_tf * num)
    return data_tf

def generate_tf_data(expression_data):
    data_tf = expression_data.copy()
    for i in range(data_tf.shape[0]):
        row = data_tf[i]
        row = (np.log10(row / len(row) + 10 ** -4) + 4)
        data_tf[i] = np.around(row * 20)
    return data_tf.transpose()


def generate_ntf_data(expression_data):
    data_ntf = expression_data.copy()
    for i in range(data_ntf.shape[0]):
        row = data_ntf[i]
        row = (np.log10(row / len(row) + 10 ** -4) + 4) / 4
        data_ntf[i] = np.around(row * 10)
    return data_ntf.transpose()


def generate_indel_list(length):
    test_size = int(0.1 * length)
    whole_list = range(length)
    test_list = random.sample(whole_list, test_size)
    train_list = [i for i in whole_list if i not in test_list]
    random.shuffle(test_list)
    random.shuffle(train_list)
    return train_list, test_list

def train_test_split():
    x_train = data_X[:, train_list, :]
    x_test = data_X[:, test_list, :]
    y_train = data_Y[train_list]
    y_test = data_Y[test_list]
    return x_train, x_test, y_train, y_test

def generate_multi_data(index):
    data_file_list = []
    for i in Network_list:
        if i == index:
            continue
        file_name = pre_file + "ExpressionData" + str(i) + ".csv"
        data_file_list.append(file_name)

    multi_data = []
    multi_data.append(data_tf)
    for i in range(2):
        pair_expression_data = generate_expression_data(data_file_list[i])
        pair_tf_data = generate_tf_data(pair_expression_data)
        pair_data = generate_pair_data(pair_tf_data)
        multi_data.append(pair_data)
    return np.array(multi_data)


class MyDataset():
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, item):
        feature_list = []
        for i in range(3):
            feature = self.x[i][item].clone().detach().requires_grad_(True)
            feature_list.append(feature)
        targets = self.y[item].clone().detach().requires_grad_(True)
        return torch.stack(feature_list), targets

    def __len__(self):
        return len(self.y)


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x

class Config(object):
    def __init__(self, input_size, num_classes):
        self.model_name = 'DeepGRNMS'
        self.dropout = 0.2                                           # 随机失活
        self.input_size = input_size                                 # 词表大小，在运行时赋值
        self.embedding_dim = 5                                       # 字向量维度
        self.num_filters = 8                                         # 卷积核数量(channels数)
        self.sent_len = data_tf.shape[1]
        self.n_hidden = 8
        self.num_classes = num_classes  # 类别数
        self.learning_rate = 1e-3  # 学习率


class DPCNN_COV(nn.Module):
    def __init__(self, config):
        super(DPCNN_COV, self).__init__()

        self.embedding = nn.Embedding(config.input_size, config.embedding_dim)

        # 1. pair embedding
        self.global_embedding = nn.Sequential(
            nn.Conv1d(config.embedding_dim, config.num_filters,
                      kernel_size=config.sent_len),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.out_pair = nn.Sequential(
            nn.Linear(config.num_filters, config.n_hidden),
            nn.BatchNorm1d(config.n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_hidden, config.n_hidden)
        )

        # 2. dpcnn embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(config.embedding_dim, config.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=config.num_filters),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=config.num_filters),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters,
                      kernel_size=3, padding=1),
        )

        self.num_seq = config.sent_len
        resnet_block_list = []
        while (self.num_seq > 2):
            resnet_block_list.append(ResnetBlock(config.num_filters))
            self.num_seq = self.num_seq // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.out_main = nn.Sequential(
            nn.Linear(config.num_filters * self.num_seq, config.n_hidden),
            nn.BatchNorm1d(config.n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )

        # 3. contact embedding
        self.out_fc = nn.Sequential(
            nn.Linear(config.n_hidden * 3, config.num_classes),
            nn.BatchNorm1d(config.num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.num_classes, config.num_classes)
        )

    def forward(self, X):
        output_main = self.forward_feature(X[0])
        out_pair = torch.cat([self.forward_pair_feature(X[i]) for i in range(1, 3)], 1)
        out = torch.cat((output_main, out_pair), 1)
        return self.out_fc(out)

    def forward_feature(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        x = self.conv_block(x)
        x = self.resnet_layer(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(batch_size, -1)
        return self.out_main(x)

    def forward_pair_feature(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.global_embedding(x)
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(batch_size, -1)
        return self.out_pair(x)


if __name__ == '__main__':
    Network = 1
    Network_list = [1, 2, 3]
    pre_file = "dyn-TF/"
    data_file = pre_file + "ExpressionData" + str(Network) + ".csv"
    output_file = "output/coexpressed_result" + str(Network) + ".txt"
    expression_data = generate_expression_data(data_file)
    print("The number of gene:", expression_data.shape[0])
    print("The number of sample:", expression_data.shape[1])
    data_tf = generate_tf_data(expression_data)
    data_ntf= generate_ntf_data(expression_data)
    input_size = np.max(np.max(data_tf))
    num_classes = np.max(np.max(data_ntf))
    multi_data = generate_multi_data(Network)
    train_list, test_list = generate_indel_list(expression_data.shape[1])

    print("--------------------------------Training Begin!--------------------------------------")
    coexpressed_result = np.zeros((data_ntf.shape[1], data_tf.shape[1]))
    input_size = np.int64(input_size + 1)
    num_classes = np.int64(num_classes + 1)
    for j in range(data_ntf.shape[1]):
        print('-------------------------------------', j, '--------------------------------------------')
        iterations = 500
        batch_size = 32
        data_test = data_ntf[:, [j]].copy()
        data_X = multi_data.copy()
        data_X[0, :, j] = 0
        data_medium = multi_data.copy()
        data_medium[0, :, j] = 0
        data_Y = data_test.copy()

        # print("1. 准备数据")
        x_train, x_test, y_train, y_test = train_test_split()
        train_dataset = MyDataset(x_train, y_train)
        test_dataset = MyDataset(x_test, y_test)
        train_data = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_data = DataLoader(dataset=test_dataset, batch_size=32)

        # print("2. 构建模型，损失函数和优化器")
        config = Config(input_size, num_classes)
        model = DPCNN_COV(config)
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # print("3. 开始训练")
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        for epoch in range(iterations):
            # 3.1 训练模型
            train_loss = 0
            train_acc = 0
            for i, data in enumerate(train_data):
                feature_list, targets = data
                inputs = feature_list.permute(1, 0, 2)
                targets = targets.squeeze(1)
                batch_size = inputs[0].shape[0]
                inputs = Variable(inputs)
                targets = Variable(targets)
                inputs = inputs.clone().detach().requires_grad_(True)
                targets = targets.clone().detach().requires_grad_(True)
                inputs = inputs.to(torch.long)
                targets = targets.to(torch.long)
                inputs = inputs.cuda()
                targets = targets.cuda()
                output = model(inputs)
                loss = criterion(output, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss = train_loss + loss.item()
                _, pred = torch.max(output.data, 1)
                num_correct = (pred == targets).sum().item()
                train_acc = train_acc + num_correct / batch_size
            losses.append(train_loss / len(train_data))
            acces.append(train_acc / len(train_data))

            # 3.2 评估模型
            eval_loss = 0
            eval_acc = 0
            for feature_list, targets in test_data:
                inputs = feature_list.permute(1, 0, 2)
                targets = targets.squeeze(1)
                batch_size = inputs[0].shape[0]
                inputs = Variable(inputs)
                targets = Variable(targets)
                inputs = inputs.clone().detach().requires_grad_(True)
                targets = targets.clone().detach().requires_grad_(True)
                inputs = inputs.to(torch.long)
                targets = targets.to(torch.long)
                inputs = inputs.cuda()
                targets = targets.cuda()
                output = model(inputs)
                loss = criterion(output, targets)
                eval_loss = eval_loss + loss.item()
                _, pred = torch.max(output.data, 1)
                num_correct = (pred == targets).sum().item()
                eval_acc = eval_acc + num_correct / batch_size
            eval_losses.append(eval_loss / len(test_data))
            eval_acces.append(eval_acc / len(test_data))
            if (epoch + 1) % 100 == 0:
                print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
                      .format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data),
                              eval_loss / len(test_data), eval_acc / len(test_data)))


        # print("4. 计算共表达结果")
        accu = []
        for i in range(data_tf.shape[1]):
            inputs = data_medium.copy()
            inputs[0, :, i] = 0
            targets = data_test
            batch_size = targets.shape[0]
            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            targets = targets.squeeze(1)
            inputs = Variable(inputs)
            targets = Variable(targets)
            inputs = inputs.clone().detach().requires_grad_(True)
            targets = targets.clone().detach().requires_grad_(True)
            inputs = inputs.to(torch.long)
            targets = targets.to(torch.long)
            inputs = inputs.cuda()
            targets = targets.cuda()
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            num_correct = (pred == targets).sum().item()
            accu.append(num_correct / batch_size)
        coexpressed_result[j, :] = accu
    np.savetxt(output_file, coexpressed_result)
    print("--------------------------------Training END!--------------------------------------")
