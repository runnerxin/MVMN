# =============================================
# @File     : base_matrix.py
# @Software : PyCharm
# @Time     : 2019/7/1 21:13
# =============================================

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
import pickle
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn.modules.module import Module
import torch.nn.functional as f
from datetime import datetime
import sys
sys.path.append('..')
from utils import Data, TrainDataset, TestDataset, valid_evaluate, test_evaluate


SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


class GraphAttentionLayer(Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, input_, a_hat):

        h = torch.mm(input_, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                             h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)

        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(a_hat > 0, e, zero_vec)
        attention = f.softmax(attention, dim=1)
        attention = f.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return f.elu(h_prime)


class GAT(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out, dropout, alpha, n_heads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.dropout = dropout
        self.n_heads = n_heads

        self.attentions = [GraphAttentionLayer(n_feature, n_hidden, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(n_hidden, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        #
        # 1
        temp = torch.zeros((x.shape[0], self.n_hidden)).cuda()
        for att in self.attentions:
            temp += att(x, adj)
        x = temp / self.n_heads

        x = self.out_att(x, adj)
        return x


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))       # hang
    r_inv = np.power(row_sum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    # D^(-0.5)AD^(-0.5)
    mx = np.multiply(np.multiply(r_inv, mx), r_inv)
    return mx


def get_adj(data_path_):

    data = pickle.loads(open(data_path_, 'rb').read())
    adj = np.zeros((data.user_number, data.user_number))
    for pair in data.true_train_relation:
        adj[pair[0], pair[1]] = adj[pair[1], pair[0]] = 1

    # A_hat = normalize(adj + np.eye(data.user_number))
    # A_hat = torch.FloatTensor(A_hat)
    return adj


def make_position_encoding(xp, batch, length, n_units, f=10000.):
    assert(n_units % 2 == 0)
    position_block = xp.broadcast_to(
        xp.arange(length)[None, None, :],
        (batch, n_units // 2, length)).astype('f')
    unit_block = xp.broadcast_to(
        xp.arange(n_units // 2)[None, :, None],
        (batch, n_units // 2, length)).astype('f')

    rad_block = position_block / (f * 1.) ** (unit_block / (n_units // 2))
    sin_block = xp.sin(rad_block)
    cos_block = xp.cos(rad_block)
    emb_block = xp.concatenate([sin_block, cos_block], axis=1)

    return emb_block


class MyModel(nn.Module):

    def __init__(self, location_number, time_number, user_number):
        super(MyModel, self).__init__()

        self.location_number = location_number
        self.time_number = time_number
        self.user_number = user_number
        self.embedding_dim = 64
        self.top_dim = 200

        self.location_embedding = nn.Embedding(self.location_number+1, self.embedding_dim)
        self.time_embedding = nn.Embedding(self.time_number, self.embedding_dim)
        self.time_embedding.weight.data.copy_(
            torch.from_numpy(make_position_encoding(np, 1, self.time_number, 64, 100)[0].T))
        self.time_embedding.weight.requires_grad = False

        self.hidden_size = 128
        self.num_layers = 1
        self.time_rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.time_v = Parameter(torch.Tensor(self.hidden_size, 1))
        self.time_w = Parameter(torch.Tensor(1, 1))
        self.time_b = Parameter(torch.Tensor(1, 1))
        self.time_v.data.uniform_(-0.1, 0.1).float()
        self.time_w.data.uniform_(-0.1, 0.1).float()
        self.time_b.data.uniform_(-0.1, 0.1).float()

        self.user_embedding = nn.Embedding(self.user_number, self.embedding_dim)
        self.gat = GAT(self.embedding_dim, self.embedding_dim, self.embedding_dim, 0.5, 0.2, 3)
        self.gat_embedding = None

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * self.top_dim + self.hidden_size + self.embedding_dim, 2)

    def get_time_gap_process(self, length, hidden_out, time_gap):

        time_gap = torch.cat(tensors=(time_gap[:, 1:],
                                      torch.from_numpy(np.zeros((time_gap.shape[0], 1))).float().cuda()), dim=-1)

        p1 = hidden_out.bmm(self.time_v.repeat(hidden_out.shape[0], 1, 1)).squeeze(-1)
        p2 = self.time_w * time_gap
        p3 = (1 / self.time_w) * torch.exp(p1 + self.time_b)
        p4 = (1 / self.time_w) * torch.exp(p1 + p2 + self.time_b)
        f1 = p1 + p2 + self.time_b + p3 - p4

        loss = 0
        for batch_index in range(hidden_out.shape[0]):
            loss += torch.mean(f1[batch_index, :length[batch_index]-1])

        time_loss = - loss / hidden_out.shape[0]
        return time_loss

    def get_gat(self, index, adj):
        feature = self.user_embedding(index)
        self.gat_embedding = self.gat(feature, adj)

    def forward(self, u1, u2, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2):

        location_emb1 = self.location_embedding(loc_1)
        location_emb2 = self.location_embedding(loc_2)
        x_1_norm = torch.sqrt(torch.sum(location_emb1.pow(2), dim=2))
        x_2_norm = torch.sqrt(torch.sum(location_emb2.pow(2), dim=2))
        ji = x_1_norm.view(-1, 200, 1) * x_2_norm.view(-1, 1, 200)
        cos_matrix = location_emb1.bmm(location_emb2.transpose(1, 2)) / ji

        # matrix_ans临时构造，最后去掉第一行
        matrix_ans = torch.from_numpy(np.zeros((1, self.top_dim*2))).float().cuda().view(1, -1)
        for batch_index in range(cos_matrix.shape[0]):
            # 矩阵切片
            qie_matrix = cos_matrix[batch_index, :length_1[batch_index], :]
            qie_matrix = qie_matrix[:, :length_2[batch_index]]

            hang, _ = torch.max(qie_matrix, dim=1)
            lie, _ = torch.max(qie_matrix, dim=0)
            padding = torch.from_numpy(np.zeros((1, self.top_dim))).float().cuda()
            hang = torch.cat(tensors=(hang.view(1, -1), padding), dim=1)
            lie = torch.cat(tensors=(lie.view(1, -1), padding), dim=1)

            cos_vec = torch.cat(tensors=(hang[:, :self.top_dim], lie[:, :self.top_dim]), dim=1)
            matrix_ans = torch.cat(tensors=(matrix_ans, cos_vec), dim=0)

        # outs = self.fc(self.dropout(ans[1:, :]))
        # ----------------------------------------------------------------------------

        time_emb1 = self.time_embedding(time_1)
        time_emb2 = self.time_embedding(time_2)
        hidden_out_1, state1 = self.time_rnn(self.dropout(time_emb1))
        hidden_out_2, state2 = self.time_rnn(self.dropout(time_emb2))

        # point process
        loss_1 = self.get_time_gap_process(length_1, hidden_out_1, time_gap_1)
        loss_2 = self.get_time_gap_process(length_2, hidden_out_2, time_gap_2)
        time_loss = loss_1 + loss_2
        # print(time_loss)
        # -------------------------------------------------------------
        bz_1 = (length_1 - 1).view(hidden_out_1.shape[0], 1, -1)
        bz_1 = bz_1.repeat(1, 1, 128).cuda()
        seq_output_1 = torch.gather(hidden_out_1, 1, bz_1)
        bz_2 = (length_2 - 1).view(hidden_out_2.shape[0], 1, -1)
        bz_2 = bz_2.repeat(1, 1, 128).cuda()
        seq_output_2 = torch.gather(hidden_out_2, 1, bz_2)
        time_ans = seq_output_1.squeeze(1) * seq_output_2.squeeze(1)

        # -------------------------
        u1_emb = self.gat_embedding.index_select(0, u1)
        u2_emb = self.gat_embedding.index_select(0, u2)

        u_hadamard = u1_emb * u2_emb

        outs = self.fc(self.dropout(torch.cat(tensors=(matrix_ans[1:, :],
                                                       torch.tanh(time_ans),
                                                       torch.tanh(u_hadamard)), dim=1)))
        return outs, time_loss


def train(data_path_in, lr_):

    data = pickle.loads(open(data_path_in, 'rb').read())

    # 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。
    train_dataset = TrainDataset(data)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    valid_dataset = TestDataset(data, test=False)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)

    train_adj = get_adj(data_path_in)
    user_index = torch.LongTensor([i for i in range(data.user_number)])
    user_index = user_index.cuda()
    # ----------------------------------------------------------------------------

    model = MyModel(data.location_number, 7*24, data.user_number).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)
    loss_func = nn.CrossEntropyLoss()

    print('train start')
    max_auc = 0

    for epoch in range(epoch_times):
        all_loss = 0
        model.train()
        time_start = datetime.now()
        for i, data_loader in enumerate(train_loader):

            # 将数据从 train_loader 中读出来
            u1, u2, y, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2 = data_loader
            u1, u2, y = u1.cuda(), u2.cuda(), y.cuda()
            length_1, length_2 = length_1.cuda(), length_2.cuda()
            loc_1, loc_2 = loc_1.cuda(), loc_2.cuda()
            time_1, time_2 = time_1.cuda(), time_2.cuda()
            time_gap_1, time_gap_2 = time_gap_1.cuda(), time_gap_2.cuda()

            new_adj = np.ones_like(train_adj)        # 行 和不为1
            for index_u in range(len(u1)):
                new_adj[u1[index_u], u2[index_u]] = 0
                new_adj[u2[index_u], u1[index_u]] = 0
            drop_adj = np.where(new_adj > 0, train_adj, new_adj)

            A_hat = normalize(drop_adj + np.eye(data.user_number))
            A_hat = torch.FloatTensor(A_hat).cuda()
            # print(A_hat)

            model.get_gat(user_index, A_hat)

            prediction, t_loss = model(u1, u2, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2)
            loss = loss_func(prediction, y) + 0.005 * t_loss
            all_loss += loss

            optimizer.zero_grad()       # clear gradients for this training step
            loss.backward()             # back propagation, compute gradients
            optimizer.step()
            # break

        # ------------------------------valid------------------------------
        model.eval()
        verify_auc = valid_evaluate(model, valid_loader)

        if verify_auc > max_auc:
            max_auc = verify_auc
            torch.save(model, model_save_path)

        time_end = datetime.now()
        print('epoch: ', epoch,
              'train_loss:', all_loss,
              'verify_auc:', verify_auc,
              'time:', (time_end - time_start).seconds)


def test(data_path_, model_save_path_):
    print(model_save_path_[2:-4])
    data = pickle.loads(open(data_path_, 'rb').read())
    test_dataset = TestDataset(data, test=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    model = torch.load(model_save_path).cuda()
    test_evaluate(model, test_loader)


if __name__ == '__main__':
    data_path = '../data/gowalla_data_4.pkl'
    model_save_path = './l2.pkl'
    epoch_times = 200
    learn_rating = 1e-4

    train(data_path, learn_rating)
    test(data_path, model_save_path)
