## 在指定目录下创建虚拟环境

```
conda create --prefix=/your/path/env/gmsf python=3.8	# 在/your/path/env/目录下创建名为gmsf的虚拟环境，指定python编译器版本为3.8，不推荐3.9及以上版本，
```

**激活命令**

```
conda activate /your/path/env/gmsf
```

**退出命令**

```
conda deactivate
```

## 依赖项安装

### torch安装

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

**验证是否可以调用GPU**

```
torch.cuda.is_available()
```

获取一下几个特定安装包（可能会出现获取失败以及下载到一半中断的问题，多试几次就好）

友情提示：速度可能很慢，保持耐心

<!-- pytorch与cuda所对应的各版本一些特殊依赖包网址：https://data.pyg.org/whl/ -->

```
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
```

**安装**

```
pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl 
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
```

### 其他依赖项安装

<!--国内常用镜像源：清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/
中国科学技术大学：https://pypi.mirrors.ustc.edu.cn/simple
豆瓣：http://pypi.douban.com/simple/
阿里云：http://mirrors.aliyun.com/pypi/simple/
华中科技大学：http://pypi.hustunique.com/
山东理工大学：http://pypi.sdutlinux.org/
百度：https://simple.baidu.com/pypi/simple
网易云：https://mirrors.163.com/pypi/simple -->

不推荐将镜像源进行全局配置，部分包国内镜像源获取有问题

带 * 的依赖项会在安装rdkit时重新安装，在本阶段可选择性安装 

```
pip install biopython -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install boto3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install Bottleneck -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install filelock -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install cycler -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install fonttools -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install greenlet -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install Jinja2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install joblib -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install kiwisolver -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install lmdb -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install munkres -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install networkx -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install numexpr -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pycairo -i https://pypi.tuna.tsinghua.edu.cn/simple/
conda install pycairo	# 采用pip直接安装会失败
pip install pyDeprecate -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install PySnooper -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install python-dateutil -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install reportlab -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install sip -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install SQLAlchemy -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tape-proteins -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch-geometric -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torchmetrics -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install TorchSnooper -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torchsummary -i https://pypi.tuna.tsinghua.edu.cn/simple/
*pip install tornado -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### rdkit安装

1. 安装命令

   以下两个命令可以相互替换使用，一个不行就试另外一个

```
conda install -c rdkit rdkit	# 直接从rdkit官方获取，实测会报错，与其他依赖项版本无法对应
conda install -c conda-forge rdkit	# 获取conda最新rdkit安装包
```

2. rdkit安装测试

**激活环境配置所在环境**

`conda activate /lex/zhengchen/env/gmsf `

**进入python环境**

`python`

**测试代码**

```python
from rdkit import Chem
from rdkit.Chem import Draw
smi = 'CCCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4'
m = Chem.MolFromSmiles(smi)
Draw.MolToImageFile(m,"mol.jpg")
```

在当前文件夹下生成一个mol.jpg图片即表示成功
## 程序运行

### 数据准备

```
python /code/getdata.py --input_file=./test.sdf --output_file=./output.csv
```

### 亲和力预测

```
python predictD.py --mac_batch=10240 --mic_batch=256 --input_file=./test.csv --output_file=./testi.csv	# 批量预测
python predict_item.py --input_file=./test.csv --output_file=./output.csv		# 逐条预测
```

### SMILES转化学式

```
python smiles2formula.py --input_file=./test.csv --output_file=./output.csv
```

