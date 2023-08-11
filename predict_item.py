import argparse
import pandas as pd
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from tqdm import tqdm
import sys, os
import torch
from model import Model
import datetime
import csv
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import warnings
warnings.filterwarnings("ignore")

def logger(function):
    def wrapper(*args, **kwargs):
        """Record the start and end time of the function and the running time of the function"""
        start = datetime.datetime.now()
        print(f"----- {function.__name__}: start -----{start}")
        output = function(*args, **kwargs)
        print(
            f"----- {function.__name__}: end -----{datetime.datetime.now()}\n-----cost time: {datetime.datetime.now() - start}")
        return output

    return wrapper


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# 查找原子是否在允许的字符集中，若不在则为'Unknown'，返回one-hot编码格式的列表
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()  # 获取所有原子数目

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    tmp = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        tmp.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def getSmilegraph(compound_iso_smiles):
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    return smile_graph


def formatSmiles(smileslist):
    formatSimle = []
    for simles in smileslist:
        formatSimle.append(Chem.MolToSmiles(Chem.MolFromSmiles(simles), isomericSmiles=True))
    return formatSimle


def createData(smi, target_name):
    PROTEIN_FILE = script_directory + "/data/" + target_name + ".npz"
    protein = np.load(PROTEIN_FILE)['arr_0'].astype('float16')

    test_drugs = [smi]
    test_Y = [-100 for _ in test_drugs]
    test_prots = [protein for _ in test_drugs]
    test_drugs, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_prots), np.asarray(test_Y)

    compound_iso_smiles = []
    compound_iso_smiles.extend(test_drugs)
    test_drugs = formatSmiles(test_drugs)
    compound_iso_smiles = formatSmiles(compound_iso_smiles)
    smile_graph = getSmilegraph(compound_iso_smiles)
    test_data = TestbedDataset(root='data', dataset=target_name + '_test', xd=test_drugs, xt=test_prots, y=test_Y,
                               smile_graph=smile_graph)
    return test_data


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    # return output.numpy()
    return total_preds.numpy().flatten()

# @logger
def MiFunc(smi, target_name, device):
    test_data = createData(smi, target_name)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    model = Model().to(device)
    path_checkpoint = script_directory + "/model/model.model"  # 模型加载路径
    checkpoint = torch.load(path_checkpoint)  # 加载模型
    model.load_state_dict(checkpoint)  # 加载模型可学习参数
    P = predicting(model, device, test_loader)[0]
    return P

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compute molecule affinity')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file path, please provide your data path(absolute or relative path), the file is a CSV file, and the required header is "Smiles"')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path, please provide your data path(absolute or relative path), the file is a CSV file')
    # Micro
    args = parser.parse_args()
    cuda_name = "cuda:0"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    target_name = '5vd0'
    # df = pd.read_csv("./data/Smiles800W.csv")
    # smiles_list = df["Smiles"].tolist()
    # smiles = 'CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1'
    input_file_name = args.input_file
    assert input_file_name.endswith(".csv"), "Please check your input, end with '.csv'."
    df = pd.read_csv(input_file_name)
    smiles_list = list(df['Smiles'])
    # smiles_list = ["CCC", "CCCCC", "COCCC"]
    in_l = []
    out_l = []
    for item in smiles_list:
        try:
            P = MiFunc(item, target_name, device)
            print(P)
            in_l.append(item)
            out_l.append(P)
            time.sleep(0.3)
        except:
            print("SMILES ERROR")
            continue
    result = pd.DataFrame()
    result['Smiles'] = in_l
    result['affinity'] = out_l
    output_file_path = args.output_file
    assert output_file_path.endswith(".csv"), "Please check your output, end with '.csv'."
    result.to_csv(output_file_path, index=False)