import argparse
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd
from tqdm import tqdm

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

# 获取当前脚本所在文件夹的绝对路径
script_directory = os.path.dirname(script_path)

def smiles_to_formula(smiles):
    # 从SMILES字符串创建分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 计算分子的分子式
    formula = rdMolDescriptors.CalcMolFormula(mol)

    return formula

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transforming Molecular SMILES into Chemical Expressions')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file path, please provide your data path(absolute or relative path), the file is a CSV file, and the required header is "Smiles"')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path, please provide your data path(absolute or relative path), the file is a CSV file')
    # Micro
    args = parser.parse_args()
    input_file_name = args.input_file
    assert input_file_name.endswith(".csv"), "Please check your input, end with '.csv'."
    df = pd.read_csv(input_file_name)
    smiles = df["Smiles"].tolist()
    formulas = []
    for i in tqdm(range(len(smiles))):
        smile = smiles[i]
        formula = smiles_to_formula(smile)
        # print(formula)  # C2H6O
        formulas.append(formula)
    df["formula"] = formulas
    output_file_path = args.output_file
    assert output_file_path.endswith(".csv"), "Please check your output, end with '.csv'."
    df.to_csv(output_file_path, index=False)