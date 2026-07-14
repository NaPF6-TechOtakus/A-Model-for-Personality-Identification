from transformers import BertTokenizer,BertModel
import torch.nn as nn

Path=r"D:\VScode\Personality Identification\config"    #r表示原始字符串，不转义特殊字符

tokenizer=BertTokenizer.from_pretrained(Path)   #读取配置文件
model=BertModel.from_pretrained(Path)   #引入分词器和Bert模型

#提取CLS向量
text="I love playing Genshin."
encoded = tokenizer(text,return_tensors='pt')
CSL_vector=model(**encoded).last_hidden_state[:,0,:]    #1条样本，8 个 token，每个 token 768 维

#CLS向量经过三层全连接层，输出五维人格向量

fc=nn.Sequential(#nn是神经网络
    nn.Linear(768,256), #[256,768]*[768,1]=[256,1]  #Linear线性变换
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,5)
)
ocean=fc(CSL_vector)
print(ocean)