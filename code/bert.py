from transformers import BertTokenizer, BertModel

MODEL_PATH = r"D:\VScode\Personality Identification\config"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
model.eval()    #评估模式

text = "I love playing Genshin."
encoded_input = tokenizer(text, return_tensors='pt')    #pytorch 张量，一种夯爆的数组
output = model(**encoded_input)
print("last_hidden_state 形状:", output.last_hidden_state.shape)
print("pooler_output 形状:", output.pooler_output.shape)

cls = output.last_hidden_state[0, 0]
print("\n[CLS] 向量前768维:", [round(x.item(), 5) for x in cls[:768]])
