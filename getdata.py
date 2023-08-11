from rdkit import Chem
import csv
import pandas as pd
import argparse

def read_sdf_with_properties(file_path):
    molecule_data_list = []
    suppl = Chem.SDMolSupplier(file_path)
    for molecule in suppl:
        if molecule is not None:
            mol_prop = {}
            # 获取分子的名称
            # name = molecule.GetProp("_Name")  # 使用"_Name"关键字获取名称
            smiles_sdf = Chem.MolToSmiles(molecule)
            mol_prop["trans_smiles"] = smiles_sdf
            for prop_name in molecule.GetPropNames():
                prop_value = molecule.GetProp(prop_name)
                mol_prop[prop_name] = prop_value
            molecule_data_list.append(mol_prop)

    return molecule_data_list

def get_value_by_priority(keys, my_dict):
    for key in keys:
        if key in my_dict:
            return my_dict[key]
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracting data from SDF files')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input file path, please provide your data path(absolute or relative path), the file is a SDF file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path, please provide your data path(absolute or relative path), the file is a CSV file')
    args = parser.parse_args()
    # 文件路径
    sdf_file_path = args.input_file
    # 获取SDF文件中每个分子的名称和指定属性
    molecule_data_list = read_sdf_with_properties(sdf_file_path)
    new_data = []

    for molecule in molecule_data_list:
        _item = {}
        _item["Smiles"] = molecule["trans_smiles"]
        # _item["ID"] = get_value_by_priority(["IDNUMBER", "Compound_ID", "idnumber", "ID", "id"], molecule)
        # _item["Link"] = get_value_by_priority(["Link"], molecule)
        # _item["source_id"] = get_value_by_priority(["source_id"], molecule)
        new_data.append(_item)
    # print(new_data)

    df = pd.DataFrame(new_data)

    # CSV文件路径
    csv_file_path = args.output_file

    # 写入CSV文件
    df.to_csv(csv_file_path, index=False)

    print("CSV file has been created successfully.")
