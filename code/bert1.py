from transformers import BertTokenizer, BertModel

MODEL_PATH = r"D:\VScode\Personality Identification\config"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertModel.from_pretrained(MODEL_PATH)
model.eval()

text_pos = "I love playing Genshin."
text_neg = "I hate playing Genshin."
text_neu = "I play Genshin."


out_pos = model(**tokenizer(text_pos, return_tensors='pt'))
out_neg = model(**tokenizer(text_neg, return_tensors='pt'))
out_neu = model(**tokenizer(text_neu, return_tensors='pt'))


vec_pos = out_pos.last_hidden_state[0, 0]
vec_neg = out_neg.last_hidden_state[0, 0]
vec_neu = out_neu.last_hidden_state[0, 0]

# 计算每维的差值 (绝对值)，排序找变化最大的维度
diff = (vec_pos - vec_neg).abs()           # [768] ，取绝对值
top10_indices = diff.argsort(descending=True)[:10]  # 差最大的前10个索引

print("=== 两层句子的 [CLS] 向量对比 ===")
print()
print("love 句:   I love playing Genshin.")
print("hate 句:   I hate playing Genshin.")
print("neu 句:   I play Genshin.")
print()
print(f"{'维度':<6} {'love句的值':>10} {'neu句的值':>10} {'hate句的值':>10} {'差值':>10}")
print("-" * 60)
for i in top10_indices:
    v_pos = vec_pos[i].item()
    v_neu = vec_neu[i].item()
    v_neg = vec_neg[i].item()
    diff_val = abs(v_pos - v_neg)
    print(f"dim {i:<3} {v_pos:>10.5f} {v_neu:>10.5f} {v_neg:>10.5f} {diff_val:>10.5f}")

# 再统计一下：768 维里平均变化多少
print()
print(f"平均每维变化: {diff.mean().item():.5f}")
print(f"最大变化:     {diff.max().item():.5f}  (维度 {diff.argmax().item()})")