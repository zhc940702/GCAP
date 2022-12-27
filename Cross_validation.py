import os
import torch
import random
import pickle
import argparse
import numpy as np
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from Network import ConvNCF
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from functools import partial
from sklearn.metrics import hamming_loss
import pandas as pd
def get_avg_auc_aupr(Pred_score, True_label):
    total_auc = 0
    total_aupr = 0
    for i in range(Pred_score.shape[1]):
        iprecision, irecall, ithresholds = metrics.precision_recall_curve(True_label[:, i], Pred_score[:, i], pos_label=1, sample_weight=None)
        aupr_i = metrics.auc(irecall, iprecision)
        auc_i = metrics.roc_auc_score(True_label[:, i], Pred_score[:, i])
        total_auc = total_auc + auc_i
        total_aupr = total_aupr + aupr_i
    avg_auc = total_auc / Pred_score.shape[1]
    avg_aupr = total_aupr / Pred_score.shape[1]
    return avg_auc, avg_aupr

MAX_SEQ_DRUG = 100
smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
                   '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
                   'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
                   'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
                   'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))
drug_feature_dimension = [2048, 2586, 1024, 315, 881, 200]

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1, 1)).toarray()

def read_raw_data(data_train, data_test, args):
    rawpath = args.rawpath

    gii = open(rawpath + 'side_vector_level_123.pkl', 'rb')
    s_feature1 = pickle.load(gii)
    gii.close()

    gii = open(rawpath + 'drug_side_association_matrix.pkl', 'rb')
    drug_side_association_matrix = pickle.load(gii)
    gii.close()

    gii = open(rawpath + 'drug_side_serverity_matrix.pkl', 'rb')
    drug_side_serverity_matrix = pickle.load(gii)
    gii.close()

    gii = open(rawpath + 'drug_smiles.pkl', 'rb')
    drug_smiles = pickle.load(gii)
    gii.close()

    for i in range(data_test.shape[0]):
        drug_side_association_matrix[int(data_test[i, 0]), int(data_test[i, 1])] = 0
        drug_side_serverity_matrix[int(data_test[i, 0]), int(data_test[i, 1])] = 0

    drug_features, side_features = [], []
    side_features = s_feature1

    Smiles = []
    for i in range(len(drug_smiles)):
        Smiles.append(drug_smiles[i][1].replace(" ", ""))

    drug_side_all_label = []
    drug_side_all_label.append(drug_side_association_matrix)
    drug_side_all_label.append(drug_side_serverity_matrix)

    split_smiles = []
    for i in range(len(Smiles)):
        split_smiles.append(drug_2_embed(trans_drug(Smiles[i])))
    node_featurizer = AttentiveFPAtomFeaturizer()
    edge_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
    fc = partial(smiles_to_bigraph, add_self_loop=True)
    smiles_graph = []
    for i in range(len(Smiles)):
        v_d = Smiles[i]
        v_d = fc(smiles=v_d, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
        smiles_graph.append(v_d)
    return smiles_graph, split_smiles, drug_features, side_features, drug_side_all_label

def train_test(data_train, data_test, data_neg, fold, args):

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    smiles_graph, split_smiles, drug_features, side_features, drug_side_all_label = read_raw_data(data_train, data_test, args)
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(data_train[:, 0]), torch.LongTensor(data_train[:, 1]), torch.LongTensor(data_train[:, 2]), torch.from_numpy(data_train[:, 3::]).float())
    testset = torch.utils.data.TensorDataset(torch.LongTensor(data_test[:, 0]), torch.LongTensor(data_test[:, 1]), torch.LongTensor(data_test[:, 2]), torch.from_numpy(data_test[:, 3::]))

    _train = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    drug_feature_type = drug_feature_dimension[args.drugtype]
    base_config_TextCNN = {'dropout_rate': args.droprate,
                           'use_element': 100,
                           'vocab_size': 100,
                           'embedding_size': 64,
                           'feature_size': 64,
                           'max_text_len': 100,
                           'window_sizes': [1, 3, 5, 7],
                           }
    model = ConvNCF([893, 1073], drug_feature_type, [792, 300], args.embed_dim, args.droprate, **base_config_TextCNN).to(device)
    a_criterion = nn.CrossEntropyLoss()
    s_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    test_AUC_mn, test_AUPR_mn, test_avg_auc_mn, test_avg_aupr_mn = 0, 0, 0, 0
    endure_count = 0

    start = time.time()
    loss_mn = 10000
    for epoch in range(1, args.epochs + 1):
        total_loss = train(smiles_graph, split_smiles, drug_features, side_features, drug_side_all_label, model, _train, optimizer, a_criterion, s_criterion, device)
        association_auc, association_iPR_auc, avg_auc, avg_aupr, pred1, pred2, total_preds, total_labels = model_test(smiles_graph, split_smiles, drug_features, side_features, drug_side_all_label, model, _train, _train, device)

        time_cost = time.time() - start
        print("Time: %.2f Epoch: %d <Train> A_auc: %.5f, A_aupr: %.5f, A_avg_auc: %.5f, A_avg_aupr: %.5f " % (
        time_cost, epoch, association_auc, association_iPR_auc, avg_auc, avg_aupr))

        # if total_loss < loss_mn:
        if test_AUC_mn < association_auc and test_avg_auc_mn < avg_auc:
            test_AUC_mn, test_avg_auc_mn = association_auc, avg_auc
            # loss_mn = total_loss
            t_association_auc, t_association_iPR_auc, t_avg_auc, t_avg_aupr, t_pred1, t_pred2, t_total_preds, t_total_labels = model_test(
                smiles_graph, split_smiles, drug_features, side_features, drug_side_all_label, model,
                _test,
                _test,
                device)
            endure_count = 0
        else:
            endure_count += 1
        if endure_count > 10:
            break
    print("<Test> A_auc: %.5f, A_aupr: %.5f, Avg_auc: %.5f, Avg_aupr: %.5f " % (t_association_auc, t_association_iPR_auc, t_avg_auc, t_avg_aupr))

    return t_association_auc, t_association_iPR_auc, t_avg_auc, t_avg_aupr

def train(smiles_graph, split_smiles, drug_features, side_features, label_information, model, train_loader, optimizer, lossfunction1, lossfunction2, device):

    model.train()
    avg_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        index_drug, index_side, batch_a, batch_s = data
        optimizer.zero_grad()
        # index_u = batch_u.cpu().numpy()
        # index_i = batch_i.cpu().numpy()
        one_label_index_1 = np.nonzero(batch_a.data.numpy())

        index_drug = index_drug.numpy()
        index_side = index_side.numpy()

        batch_smiles_graph = np.array(smiles_graph)[index_drug]
        batch_split_smiles = np.array(split_smiles)[index_drug]
        # batch_drug_features = drug_features[index_drug]
        # batch_side_feature1 = np.array(side_feature1)[index_side]
        # batch_side_feature2 = np.array(side_feature2)[index_side]

        batch_side_feature = np.array(side_features)[index_side]

        batch_split_smiles = torch.FloatTensor(batch_split_smiles)
        # batch_drug_features = torch.FloatTensor(batch_drug_features)
        batch_side_feature1 = torch.FloatTensor(batch_side_feature)
        # batch_side_feature2 = torch.FloatTensor(batch_side_feature2)

        batch_drug_side_label_1_drug = np.array(label_information[0])[index_drug]
        batch_drug_side_label_2_drug = np.array(label_information[1])[index_drug]

        batch_drug_side_label_1_side = np.array(label_information[0]).T[index_side]
        batch_drug_side_label_2_side = np.array(label_information[1]).T[index_side]

        batch_drug_label = []
        batch_drug_label.append(torch.FloatTensor(batch_drug_side_label_1_drug))
        batch_drug_label.append(torch.FloatTensor(batch_drug_side_label_2_drug))

        batch_side_label = []
        batch_side_label.append(torch.FloatTensor(batch_drug_side_label_1_side))
        batch_side_label.append(torch.FloatTensor(batch_drug_side_label_2_side))

        batch_side_feature2 = []
        batch_drug_features = []
        a_score, s_score = model(batch_smiles_graph, batch_split_smiles, batch_drug_features, batch_side_feature1, batch_side_feature2, batch_drug_label, batch_side_label, device)

        loss1 = lossfunction1(a_score, batch_a.to(device))

        loss2 = lossfunction2(s_score[one_label_index_1], batch_s[one_label_index_1].to(device))
        # total_loss = loss1
        # print(loss3)
        # weight_loss = [0.5, 0.5]
        total_loss = loss2 + loss1
        total_loss.backward(retain_graph=True)
        optimizer.step()

        avg_loss += total_loss.item()

    return avg_loss

def model_test(smiles_graph, split_smiles, drug_features, side_features, label_information, model, test_loader, neg_loader, device):
    model.eval()
    pred1 = []
    pred2 = []
    s_truth = []
    a_truth = []
    labels_all = []
    feature_split_smiles_drug = []
    feature_vector_side = []
    feature_graph_drug = []
    predicted_all = []
    for index_drug, index_side, test_a, test_s in test_loader:
        one_label_index_1 = np.nonzero(test_a.data.numpy())
        index_drug = index_drug.numpy()
        index_side = index_side.numpy()
        test_smiles_graph = np.array(smiles_graph)[index_drug]
        test_split_smiles = np.array(split_smiles)[index_drug]
        test_side_feature1 = np.array(side_features)[index_side]
        test_split_smiles = torch.FloatTensor(test_split_smiles)
        test_side_feature1 = torch.FloatTensor(test_side_feature1)
        batch_drug_side_label_1_drug = np.array(label_information[0])[index_drug]
        batch_drug_side_label_2_drug = np.array(label_information[1])[index_drug]
        batch_drug_side_label_1_side = np.array(label_information[0].T)[index_side]
        batch_drug_side_label_2_side = np.array(label_information[1].T)[index_side]
        batch_drug_label = []
        batch_drug_label.append(torch.FloatTensor(batch_drug_side_label_1_drug))
        batch_drug_label.append(torch.FloatTensor(batch_drug_side_label_2_drug))
        batch_side_label = []
        batch_side_label.append(torch.FloatTensor(batch_drug_side_label_1_side))
        batch_side_label.append(torch.FloatTensor(batch_drug_side_label_2_side))
        feature_graph_drug.append(test_smiles_graph)
        feature_split_smiles_drug.append(test_split_smiles.data.cpu())
        feature_vector_side.append(list(test_side_feature1.data.cpu().numpy()))
        a_truth.append(list(test_a.data.cpu().numpy()))
        s_truth.append(list(test_s[one_label_index_1].data.cpu().numpy()))
        labels_all.append(test_s[one_label_index_1].data.cpu().numpy())
        test_split_simles, test_side1 = test_split_smiles.to(device), test_side_feature1.to(device)
        test_drug = []
        test_side2 = []
        scores_one, scores_two = model(test_smiles_graph, test_split_simles, test_drug, test_side1, test_side2, batch_drug_label, batch_side_label, device)
        pred1.append(list(scores_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data[one_label_index_1].cpu().numpy()))

        if device.type == 'cuda':
            predicted = scores_two[one_label_index_1].cpu().detach()
        else:
            predicted = scores_two[one_label_index_1].detach()
        predicted = predicted.numpy()
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        predicted = predicted.astype(int)
        predicted_all.append(predicted)

    pred1 = np.array(sum(pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    s_truth = np.array(sum(s_truth, []), dtype=np.float32)
    a_truth = np.array(sum(a_truth, []), dtype=np.float32)
    iprecision, irecall, ithresholds = metrics.precision_recall_curve(a_truth,
                                                                      pred1[:, 1],
                                                                      pos_label=1,
                                                                      sample_weight=None)
    association_iPR_auc = metrics.auc(irecall, iprecision)
    association_auc = metrics.roc_auc_score(a_truth, pred1[:, 1])
    total_labels = np.concatenate(labels_all)
    total_preds = np.concatenate(predicted_all)
    avg_auc, avg_aupr = get_avg_auc_aupr(pred2, total_labels)
    print('avg_auc_aupr:', avg_auc, avg_aupr)
    return association_auc, association_iPR_auc, avg_auc, avg_aupr, pred1, pred2, total_preds, total_labels

def ten_fold(args):
    rawpath = args.rawpath
    pkl_file = open(rawpath + 'final_sample.pkl', 'rb')
    final_sample = pickle.load(pkl_file)

    final_sample = np.array(final_sample).astype(int)

    X = final_sample[:, 0::]
    data = X
    data_x = []
    data_y = []

    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((X[i, 2]))
    fold = 1
    kfold = StratifiedKFold(10, random_state=10, shuffle=True)
    total_a_auc, total_a_pr_auc, total_Avg_auc, total_Avg_aupr = [], [], [], []
    data_neg = 0
    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))

        A_AUC, A_AUPR, Avg_auc, Avg_aupr = train_test(data[train].tolist(), data[test].tolist(), data_neg, fold, args)

        total_a_auc.append(A_AUC)
        total_a_pr_auc.append(A_AUPR)
        total_Avg_auc.append(Avg_auc)
        total_Avg_aupr.append(Avg_aupr)

        print("==================================fold {} end".format(fold))
        fold += 1
        print('Total_a_AUC:')
        print(np.mean(total_a_auc))
        print('Total_a_AUPR:')
        print(np.mean(total_a_pr_auc))
        print('Total_Avg_auc:')
        print(np.mean(total_Avg_auc))
        print('Total_Avg_aupr:')
        print(np.mean(total_Avg_aupr))
        sys.stdout.flush()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument('--epochs', type=int, default=1000,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005,
                        metavar = 'FLOAT', help='learning rate')
    # 64是L0层映射出的维度数
    parser.add_argument('--embed_dim', type=int, default=64,
                        metavar='N', help='embedding dimension')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        metavar='FLOAT', help='weight decay')
    parser.add_argument('--droprate', type=float, default=0.3,
                        metavar = 'FLOAT', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default = 128,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        metavar='N', help='input batch size for testing')
    parser.add_argument('--drugtype', type=int, default='0',
                        metavar='STRING', help='drug_type')
    parser.add_argument('--sidetype', type=int, default='0',
                        metavar='STRING', help='side_type')
    parser.add_argument('--rawpath', type=str, default='/data/',
                        metavar='STRING', help='rawpath')
    args = parser.parse_args()

    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))
    print('drug feature type: ' + str(args.drugtype))
    print('side feature type: ' + str(args.sidetype))
    ten_fold(args)

if __name__ == "__main__":
    main()