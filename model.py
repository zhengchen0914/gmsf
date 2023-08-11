import torch
import torch.nn as nn
import torchsnooper
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# GCN-CNN based model

# @torchsnooper.snoop()
class Model(torch.nn.Module):
    """

    """

    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=1562, num_channel_xd=200, num_channel_xt=400
                 , output_dim=128, dropout=0.5):
        super(Model, self).__init__()
        # self.custom_smiles_dict = '/lex/zhengchen/graphBertBAP/Smiles_Custom_Dict.txt'
        self.num_channel_xd = num_channel_xd
        self.n_output = n_output
        self.num_channel_xt = num_channel_xt
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.n_hidden = 64

        # Graph module
        self.gat = GATConv(num_features_xd, num_features_xd, heads=10)
        self.gcn1 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.gcn2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.gcn3 = GCNConv(num_features_xd * 10, num_features_xd * 10)

        self.fusion_fc = torch.nn.Linear(num_features_xd * 10, num_features_xd * 10)
        self.fc_g11 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g12 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g13 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)

        self.fc_g21 = torch.nn.Linear(1500, output_dim)
        self.fc_g22 = torch.nn.Linear(1500, output_dim)
        self.fc_g23 = torch.nn.Linear(1500, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1_xt = nn.Linear(num_features_xt, output_dim)

        # word embedding
        self.embedding = torch.nn.Embedding(85, num_channel_xd)

        self.bilstm = nn.LSTM(num_channel_xd, self.n_hidden, 1, bidirectional=True)
        self.lstm = nn.LSTM(2 * self.n_hidden, 2 * self.n_hidden, 1, bidirectional=False)

        self.aap_xd = nn.AdaptiveAvgPool2d((1, num_features_xd))

        self.w = nn.Parameter(torch.ones(3))

        # combined layers

        self.fc1 = nn.Linear(num_features_xd + output_dim + num_features_xt, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        gat_layer = self.relu(self.gat(x, edge_index))
        x0 = gat_layer

        # P1
        gcn_layer1 = self.relu(self.gcn1(gat_layer, edge_index))
        x1 = gcn_layer1
        resgcn_layer1 = gcn_layer1 + x0
        p1 = self.relu(self.fc_g11(torch.cat([gmp(resgcn_layer1, batch), gap(resgcn_layer1, batch)], dim=1)))
        p1 = self.dropout(p1)
        p1 = self.fc_g21(p1)

        # P2
        gcn_layer2 = self.relu(self.gcn2(resgcn_layer1, edge_index))
        x2 = gcn_layer2
        resgcn_layer2 = gcn_layer2 + x1
        p2 = torch.cat([gmp(resgcn_layer2, batch), gap(resgcn_layer2, batch)], dim=1)
        p2 = self.relu(self.fc_g12(p2))
        p2 = self.dropout(p2)
        p2 = self.fc_g22(p2)

        # P3
        gcn_layer3 = self.relu(self.gcn3(resgcn_layer2, edge_index))
        resgcn_layer3 = gcn_layer3 + x2
        fusion = self.relu(self.fusion_fc(resgcn_layer3))
        # apply global max pooling (gmp) and global mean pooling (gap)
        p3 = torch.cat([gmp(fusion, batch), gap(fusion, batch)], dim=1)
        p3 = self.relu(self.fc_g13(p3))
        p3 = self.dropout(p3)
        p3 = self.fc_g23(p3)

        # 归一化不同层权重
        w11 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w12 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w13 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))

        p3 = p1 * w11 + p2 * w12 + p3 * w13

        compound = data.smile_feature
        embedding_input = self.embedding(compound)
        # xd = word_embedding
        bilstm_output, (h_n1, c_n1) = self.bilstm(embedding_input)
        lstm_output, (h_n2, c_n2) = self.lstm(bilstm_output)
        xd = lstm_output
        xd = self.aap_xd(xd)
        xd = torch.squeeze(xd, dim=1)


        xt = data.target

        # concat
        xc = torch.cat((p3, xd, xt), 1)

        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc4(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)

        return out
