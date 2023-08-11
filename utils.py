import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import re
# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

# 获取当前脚本所在文件夹的绝对路径
script_directory = os.path.dirname(script_path)

class TestbedDataset(InMemoryDataset):
    def __init__(self, root, dataset, xd=None, xt=None, y=None, smile_graph=None, max_len_c=200, transform=None,
                 pre_transform=None):

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.max_len_c = max_len_c
        self.custom_smiles_dict = script_directory + '/Smiles_Custom_Dict.txt'
        # if os.path.isfile(self.processed_paths[0]):
        #     print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
        #
        #     self.data, self.slices = torch.load(self.processed_paths[0])
        #     print('data loaded')
        # else:
        #     print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
        self.process(xd, xt, y, smile_graph)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            if (i + 1) % 10 == 0:
                print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            try:
                smile_feature = self.smiles2id_custom(smiles)
                c_size, features, edge_index = smile_graph[smiles]
                GCNData = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
                GCNData.smile_feature = torch.LongTensor([smile_feature])
                GCNData.target = torch.FloatTensor([target])
                GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
                data_list.append(GCNData)

            except:
                with open("./error_smiles.txt", mode='a', encoding='utf-8') as file:
                    file.write(smiles + '\n')
        # print(len(data_list))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def smiles2id_custom(self, smiles):
        smiles2id_dict = {}
        pattern = "(\[|]|Br?|Cl?|H|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        with open(self.custom_smiles_dict, "r") as f:
            smiles_chars_list = f.read().split("\n")
        for chars in smiles_chars_list:
            if chars != "":
                smiles2id_dict[chars] = len(smiles2id_dict)

        smiles2split = [token for token in regex.findall(smiles)]
        smiles2id = np.array([smiles2id_dict[s] for s in smiles2split])
        feature_embedding = np.zeros((self.max_len_c))
        if len(smiles2id) > self.max_len_c:
            feature_embedding = smiles2id[:self.max_len_c]
        else:
            feature_embedding[:len(smiles2id)] = smiles2id
        return feature_embedding

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

