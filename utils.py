# =============================================
# @File     : utils.py
# @Software : PyCharm
# @Time     : 2019/2/28 14:23
# =============================================

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as f
import numpy as np
from evaluate import get_hit_precision, get_map, get_recall_k, get_precision_k


class Location:
    def __init__(self, location_id, latitude, longitude, time):
        self.location_id = location_id
        self.latitude = latitude
        self.longitude = longitude
        self.time = time

    def distance_reciprocal(self, other_latitude, other_longitude):
        # 转成弧度制other_latitude, other_longitude
        x1, y1, x2, y2 = map(math.radians, [other_latitude, other_longitude, self.latitude, self.longitude])

        earth_radius = 6378137.0
        temp = math.sin((x1 - x2) / 2) ** 2 + math.cos(x1) * math.cos(x2) * math.sin((y1 - y2) / 2) ** 2
        distance = 2 * earth_radius * math.asin(math.sqrt(temp))

        return 1.0 / distance


class Data:
    def __init__(self):
        self.user_location = None
        self.user_number = None
        self.location_number = None

        # ----------------------------------
        self.relation = None

        self.true_train_relation = None
        self.true_verify_relation = None
        self.true_test_relation = None

        self.train_relation = None
        self.verify_relation = None
        self.test_relation = None

    def set_user_location(self, u_l, u_n, l_n):
        self.user_location = u_l
        self.user_number = u_n
        self.location_number = l_n

    def set_relation(self, re, ttr, tvr, tsr):
        self.relation = re
        self.true_train_relation = ttr
        self.true_verify_relation = tvr
        self.true_test_relation = tsr

    def set_true_relation(self, tr, vr, sr):
        self.train_relation = tr
        self.verify_relation = vr
        self.test_relation = sr


def sub_time(time1, time2):
    if time1 > time2:
        return (time1 - time2).seconds / 3600
    else:
        return (time2 - time1).seconds / 3600


class TrainDataset(Dataset):
    def __init__(self, pickle_data):

        self.u1, self.u2 = [], []
        self.y = []
        self.length_1, self.length_2 = [], []
        self.loc_1, self.loc_2 = [], []
        self.time_1, self.time_2 = [], []
        self.time_gap_1, self.time_gap_2 = [], []

        for pair in pickle_data.train_relation:
            x1 = pickle_data.user_location[pair[0]]
            x2 = pickle_data.user_location[pair[1]]

            length_1 = len(x1)
            length_2 = len(x2)

            temp_loc1 = [piece.location_id for piece in x1]
            temp_loc2 = [piece.location_id for piece in x2]
            temp_time1 = [(piece.time.weekday() * 24 + piece.time.hour) // 1 for piece in x1]
            temp_time2 = [(piece.time.weekday() * 24 + piece.time.hour) // 1 for piece in x2]
            temp_time_gap_1 = [sub_time(x1[i_].time, x1[i_ - 1].time) if i_ != 0 else 0.0 for i_ in range(len(x1))]
            temp_time_gap_2 = [sub_time(x2[i_].time, x2[i_ - 1].time) if i_ != 0 else 0.0 for i_ in range(len(x2))]

            # padding
            ps_x1 = 200 - length_1
            ps_x2 = 200 - length_2
            temp_loc1.extend([pickle_data.location_number] * ps_x1)
            temp_loc2.extend([pickle_data.location_number] * ps_x2)
            temp_time1.extend([0] * ps_x1)
            temp_time2.extend([0] * ps_x2)
            temp_time_gap_1.extend([0] * ps_x1)
            temp_time_gap_2.extend([0] * ps_x2)

            self.u1.append(pair[0])
            self.u2.append(pair[1])
            self.y.append(pair[2])

            self.length_1.append(length_1)
            self.length_2.append(length_2)
            self.loc_1.append(temp_loc1)
            self.loc_2.append(temp_loc2)
            self.time_1.append(temp_time1)
            self.time_2.append(temp_time2)
            self.time_gap_1.append(temp_time_gap_1)
            self.time_gap_2.append(temp_time_gap_2)

        self.u1 = torch.LongTensor(self.u1)
        self.u2 = torch.LongTensor(self.u2)
        self.y = torch.LongTensor(self.y)
        self.length_1 = torch.LongTensor(self.length_1)
        self.length_2 = torch.LongTensor(self.length_2)
        self.loc_1 = torch.LongTensor(self.loc_1)
        self.loc_2 = torch.LongTensor(self.loc_2)
        self.time_1 = torch.LongTensor(self.time_1)
        self.time_2 = torch.LongTensor(self.time_2)
        self.time_gap_1 = torch.FloatTensor(self.time_gap_1)
        self.time_gap_2 = torch.FloatTensor(self.time_gap_2)

    def __getitem__(self, index):           # 返回的是tensor
        u1_ = self.u1[index]
        u2_ = self.u2[index]
        y_ = self.y[index]

        length_1_ = self.length_1[index]
        length_2_ = self.length_2[index]
        loc_1_ = self.loc_1[index]
        loc_2_ = self.loc_2[index]
        time_1_ = self.time_1[index]
        time_2_ = self.time_2[index]
        time_gap_1_ = self.time_gap_1[index]
        time_gap_2_ = self.time_gap_2[index]
        return u1_, u2_, y_, length_1_, length_2_, loc_1_, loc_2_, time_1_, time_2_, time_gap_1_, time_gap_2_

    def __len__(self):
        return len(self.y)


class TestDataset(Dataset):
    def __init__(self, pickle_data, test=False):

        self.u1, self.u2 = [], []
        self.y = []
        self.length_1, self.length_2 = [], []
        self.loc_1, self.loc_2 = [], []
        self.time_1, self.time_2 = [], []
        self.time_gap_1, self.time_gap_2 = [], []

        if test:
            epoch_relation = pickle_data.test_relation
        else:
            epoch_relation = pickle_data.verify_relation

        for one_people in epoch_relation:
            one_people_u1, one_people_u2 = [], []
            one_people_y = []
            one_people_length_1, one_people_length_2 = [], []
            one_people_loc_1, one_people_loc_2 = [], []
            one_people_time_1, one_people_time_2 = [], []
            one_people_time_gap_1, one_people_time_gap_2 = [], []

            for pair in one_people:

                x1 = pickle_data.user_location[pair[0]]
                x2 = pickle_data.user_location[pair[1]]
                length_1 = len(x1)
                length_2 = len(x2)

                temp_loc1 = [piece.location_id for piece in x1]
                temp_loc2 = [piece.location_id for piece in x2]
                temp_time1 = [(piece.time.weekday() * 24 + piece.time.hour) // 1 for piece in x1]
                temp_time2 = [(piece.time.weekday() * 24 + piece.time.hour) // 1 for piece in x2]
                temp_time_gap_1 = [sub_time(x1[i_].time, x1[i_ - 1].time) if i_ != 0 else 0.0 for i_ in range(len(x1))]
                temp_time_gap_2 = [sub_time(x2[i_].time, x2[i_ - 1].time) if i_ != 0 else 0.0 for i_ in range(len(x2))]

                # padding
                ps_x1 = 200 - len(x1)
                ps_x2 = 200 - len(x2)
                temp_loc1.extend([pickle_data.location_number] * ps_x1)
                temp_loc2.extend([pickle_data.location_number] * ps_x2)
                temp_time1.extend([0] * ps_x1)
                temp_time2.extend([0] * ps_x2)
                temp_time_gap_1.extend([0] * ps_x1)
                temp_time_gap_2.extend([0] * ps_x2)

                # --------------------------
                one_people_u1.append(pair[0])
                one_people_u2.append(pair[1])
                one_people_y.append(pair[2])

                one_people_length_1.append(length_1)
                one_people_length_2.append(length_2)
                one_people_loc_1.append(temp_loc1)
                one_people_loc_2.append(temp_loc2)
                one_people_time_1.append(temp_time1)
                one_people_time_2.append(temp_time2)
                one_people_time_gap_1.append(temp_time_gap_1)
                one_people_time_gap_2.append(temp_time_gap_2)

            self.u1.append(torch.LongTensor(one_people_u1))
            self.u2.append(torch.LongTensor(one_people_u2))
            self.y.append(torch.LongTensor(one_people_y))
            self.length_1.append(torch.LongTensor(one_people_length_1))
            self.length_2.append(torch.LongTensor(one_people_length_2))
            self.loc_1.append(torch.LongTensor(one_people_loc_1))
            self.loc_2.append(torch.LongTensor(one_people_loc_2))
            self.time_1.append(torch.LongTensor(one_people_time_1))
            self.time_2.append(torch.LongTensor(one_people_time_2))
            self.time_gap_1.append(torch.FloatTensor(one_people_time_gap_1))
            self.time_gap_2.append(torch.FloatTensor(one_people_time_gap_2))

    def __getitem__(self, index):  # 返回的是tensor
        u1_ = self.u1[index]
        u2_ = self.u2[index]
        y_ = self.y[index]

        length_1_ = self.length_1[index]
        length_2_ = self.length_2[index]
        loc_1_ = self.loc_1[index]
        loc_2_ = self.loc_2[index]
        time_1_ = self.time_1[index]
        time_2_ = self.time_2[index]
        time_gap_1_ = self.time_gap_1[index]
        time_gap_2_ = self.time_gap_2[index]
        return u1_, u2_, y_, length_1_, length_2_, loc_1_, loc_2_, time_1_, time_2_, time_gap_1_, time_gap_2_

    def __len__(self):
        return len(self.y)


def valid_evaluate(model, data_loader):
    auc = []
    model.eval()
    for i, data in enumerate(data_loader):
        u1, u2, y, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2 = data
        u1, u2, y = u1.squeeze(0).cuda(), u2.squeeze(0).cuda(), y.squeeze(0).cuda()
        length_1, length_2 = length_1.squeeze(0).cuda(), length_2.squeeze(0).cuda()
        loc_1, loc_2 = loc_1.squeeze(0).cuda(), loc_2.squeeze(0).cuda()
        time_1, time_2 = time_1.squeeze(0).cuda(), time_2.squeeze(0).cuda()
        time_gap_1, time_gap_2 = time_gap_1.squeeze(0).cuda(), time_gap_2.squeeze(0).cuda()

        prediction, _ = model(u1, u2, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2)
        prediction = f.softmax(prediction, dim=1)
        score = prediction.data[:, 1].cpu().numpy()
        label = y.data[:].cpu().numpy()

        auc.append(roc_auc_score(y_true=label, y_score=score))

    return np.mean(auc)


def test_evaluate(model, data_loader):
    auc = []
    hit_precision = []
    map_ = []
    recall_1, precision_1 = [], []
    recall_5, precision_5 = [], []
    recall_10, precision_10 = [], []

    model.eval()
    for i, data in enumerate(data_loader):
        u1, u2, y, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2 = data
        u1, u2, y = u1.squeeze(0).cuda(), u2.squeeze(0).cuda(), y.squeeze(0).cuda()
        length_1, length_2 = length_1.squeeze(0).cuda(), length_2.squeeze(0).cuda()
        loc_1, loc_2 = loc_1.squeeze(0).cuda(), loc_2.squeeze(0).cuda()
        time_1, time_2 = time_1.squeeze(0).cuda(), time_2.squeeze(0).cuda()
        time_gap_1, time_gap_2 = time_gap_1.squeeze(0).cuda(), time_gap_2.squeeze(0).cuda()

        prediction, _ = model(u1, u2, length_1, length_2, loc_1, loc_2, time_1, time_2, time_gap_1, time_gap_2)
        prediction = f.softmax(prediction, dim=1)
        score = prediction.data[:, 1].cpu().numpy()
        label = y.data[:].cpu().numpy()
        score_rank = sorted(zip(label, score), key=lambda cus: cus[1], reverse=True)

        auc.append(roc_auc_score(y_true=label, y_score=score))
        hit_precision.append(get_hit_precision(label_score_sorted_pair=score_rank))
        map_.append(get_map(label_score_sorted_pair=score_rank))
        recall_1.append(get_recall_k(label_score_sorted_pair=score_rank, k=1))
        precision_1.append(get_precision_k(label_score_sorted_pair=score_rank, k=1))
        recall_5.append(get_recall_k(label_score_sorted_pair=score_rank, k=5))
        precision_5.append(get_precision_k(label_score_sorted_pair=score_rank, k=5))
        recall_10.append(get_recall_k(label_score_sorted_pair=score_rank, k=10))
        precision_10.append(get_precision_k(label_score_sorted_pair=score_rank, k=10))

    print('----------------------------------test--------------------------------------')
    print('AUC: ', np.mean(auc))
    print('Hit_precision: ', np.mean(hit_precision))
    print('MAP: ', np.mean(map_))
    print('recall@1:  ', '%0.10f' % np.mean(recall_1), '  precision@1:  ', '%0.10f' % np.mean(precision_1))
    print('recall@5:  ', '%0.10f' % np.mean(recall_5), '  precision@5:  ', '%0.10f' % np.mean(precision_5))
    print('recall@10: ', '%0.10f' % np.mean(recall_10), '  precision@10: ', '%0.10f' % np.mean(precision_10))
