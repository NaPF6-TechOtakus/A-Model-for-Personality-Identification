import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from convokit import Corpus
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# 固定随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"使用设备: {device}  ({'GPU' if torch.cuda.is_available() else 'CPU'})")

# ============================================================
# Step 1: 数据预处理 —— 从 PSG 提取 (对话文本, OCEAN标签)
# ============================================================
print("=" * 60)
print("Step 1: 加载 PSG 数据集并提取数据...")
print("=" * 60)

corpus = Corpus(filename=r"C:\Users\loveFurina\.convokit\saved-corpora\persuasionforgood-corpus")
OCEAN_KEYS = ['open', 'conscientious', 'extrovert', 'agreeable', 'neurotic']

data = []  # [(utterances_list, [O,C,E,A,N]), ...]

for conv in corpus.iter_conversations():
    # 找到 EE (被说服者)
    ee_id = None
    all_speaker_ids_ordered = []
    seen = set()
    for utt in conv.iter_utterances():
        sid = utt.speaker.id
        if sid not in seen:
            seen.add(sid)
            all_speaker_ids_ordered.append(sid)

    if len(all_speaker_ids_ordered) == 0:
        continue

    for sid in all_speaker_ids_ordered:
        spk = corpus.get_speaker(sid)
        if all(k in spk.meta for k in OCEAN_KEYS):
            ee_id = sid
            break

    if ee_id is None:
        continue

    ee = corpus.get_speaker(ee_id)

    # 收集 EE 说的话 —— 单独保存，不拼接
    ee_utterances = []
    for utt in conv.iter_utterances():
        if utt.speaker.id == ee_id:
            ee_utterances.append(utt.text)

    if len(ee_utterances) == 0:
        continue

    ocean_label = [ee.meta[k] for k in OCEAN_KEYS]

    # 跳过 NaN 标签
    if any(v is None or (isinstance(v, float) and (v != v)) for v in ocean_label):
        continue

    # 存入: (utterances列表, OCEAN标签)
    # 与之前不同——现在保留多条句子，不拼接
    data.append((ee_utterances, ocean_label))

print(f"提取完成: {len(data)} 条 (对话, OCEAN标签) 对")

# 随机切分 (论文采用 train_test_split 保证分布一致)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f"训练集: {len(train_data)} 条 | 测试集: {len(test_data)} 条")

# 展示一条样例
print(f"\n样例对话 (前2句): {train_data[0][0][:2]}")
print(f"样例 OCEAN 标签: O={train_data[0][1][0]:.1f}  C={train_data[0][1][1]:.1f}  E={train_data[0][1][2]:.1f}  A={train_data[0][1][3]:.1f}  N={train_data[0][1][4]:.1f}")

# ============================================================
# Step 2: 加载 BERT 模型（冻结）
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 加载 BERT 模型...")
print("=" * 60)

MODEL_PATH = r"D:\VScode\Personality Identification\config"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert = BertModel.from_pretrained(MODEL_PATH)
bert = bert.to(device)  # 搬到 GPU
bert.eval()

for param in bert.parameters():
    param.requires_grad = False

print("BERT 加载完成，参数已冻结，已搬到 GPU。")

# ============================================================
# Step 3: 预处理 —— 每句话单独编码，保留序列结构
#           之前: full_text → 一个 [CLS]
#           现在: 每句话 → 一个 [CLS]，组成 (seq_len, 768) 序列
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 预处理 —— 每句话独立编码成 [CLS] 向量序列...")
print("=" * 60)

def encode_utterances(utterances_list):
    vecs = []
    with torch.no_grad():
        for text in utterances_list:
            encoded = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
            # 🔥 必须搬到 GPU——tokenizer 永远输出 CPU tensor，但 bert 在 GPU 上
            encoded = {k: v.to(device) for k, v in encoded.items()}
            cls_vec = bert(**encoded).last_hidden_state[:, 0, :]  # (1, 768)
            vecs.append(cls_vec)
    return torch.cat(vecs, dim=0).to(device)  # (seq_len, 768) → GPU


class PSGDataset(Dataset):
    """自定义 Dataset: 每条数据 = (N句话的768维序列, 5维OCEAN标签)"""
    def __init__(self, data_list):
        self.samples = []
        for utterances, label in data_list:
            seq = encode_utterances(utterances)  # (seq_len, 768)
            y = torch.tensor(label, dtype=torch.float32)
            self.samples.append((seq, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)
    # 记录每条序列的真实长度，供 Transformer 的 key_padding_mask 使用
    lengths = torch.tensor([s.shape[0] for s in seqs])
    # pad_sequence: 把变长序列补到 batch 内最大长度
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)  # (batch, max_seq_len, 768)
    labels = torch.stack(labels, dim=0)  # (batch, 5)
    return seqs_padded, labels, lengths

# 创建 Dataset 和 DataLoader
train_dataset = PSGDataset(train_data)
test_dataset  = PSGDataset(test_data)

BATCH_SIZE = 64  # 论文参数
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"训练集: {len(train_dataset)} 条 | 测试集: {len(test_dataset)} 条")
print(f"Batch size: {BATCH_SIZE} | 每个 epoch 更新: {len(train_loader)} 次")

# 看一眼数据形状
sample_seq, sample_label = train_dataset[0]
print(f"\n一条数据形状: 序列 = {sample_seq.shape}  (句子数, 768维) | 标签 = {sample_label.shape}")

# ============================================================
# Step 4: 定义 DPPR 模型（Transformer Encoder + FC）
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 定义 DPPR 模型 (Transformer Encoder + FC)...")
print("=" * 60)

class DPPR(nn.Module):   
    def __init__(self):
        super().__init__()
        # Transformer Encoder (论文参数: d_model=768, FFN_dim=1024, nhead=8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,       # PyTorch 1.10+ 支持 batch_first=True
            activation='gelu',      # GELU 激活，比 ReLU 更平滑
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # FC 回归头 (3层)
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, x, lengths=None):
        # 构造 padding mask: True = 是 padding 不用关注
        if lengths is not None:
            mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            # mask 形状: (batch, max_seq_len)  — True 表示该位置是 padding
        else:
            mask = None

        # Transformer Encoder: Self-Attention 让每句话"看"其他句子
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, max_seq_len, 768)

        # Mean Pooling: 对所有句子取平均（去掉 padding 位置）
        if mask is not None:
            # mask: True = padding → 用 ~mask 把 padding 位置的权重设为 0
            mask_expanded = (~mask).float().unsqueeze(-1)   # (batch, max_seq_len, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (batch, 768)

        # FC 回归头: 768 → 256 → 128 → 5
        return self.fc(x)


model = DPPR()
model = model.to(device)  # 搬到 GPU
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DPPR 可训练参数总数: {total_params:,}")
# Transformer Encoder: ~7.9M + FC: ~230k ≈ 8.1M

# 损失函数
loss_fn = nn.MSELoss()

# 看一下初始预测
print("\n训练前 — 前3条样本的随机预测 vs 真实标签:")
model.eval()
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    seqs, labels, lengths = sample_batch
    seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
    pred_init = model(seqs, lengths)
    for i in range(min(3, pred_init.shape[0])):
        print(f"  预测: O={pred_init[i,0]:.2f} C={pred_init[i,1]:.2f} E={pred_init[i,2]:.2f} A={pred_init[i,3]:.2f} N={pred_init[i,4]:.2f}")
        print(f"  真实: O={labels[i,0]:.1f} C={labels[i,1]:.1f} E={labels[i,2]:.1f} A={labels[i,3]:.1f} N={labels[i,4]:.1f}")

# ============================================================
# Step 5: 训练循环 (mini-batch, lr=1e-4, epochs=100)
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 开始训练...")
print("=" * 60)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 论文 lr=1e-4
epochs = 100                                                   # 论文 epochs=100

history = {
    'epoch': [],
    'train_loss': [],
    'test_loss': [],
}

for epoch in range(epochs):
    # ----- 训练阶段 -----
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        seqs, labels, lengths = batch
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        pred = model(seqs, lengths)
        loss = loss_fn(pred, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ----- 评估阶段 -----
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            seqs, labels, lengths = batch
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
            pred = model(seqs, lengths)
            loss = loss_fn(pred, labels)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)

    # ----- 记录 -----
    if epoch % 10 == 0:
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        print(f"Epoch {epoch:3d} | 训练Loss: {avg_train_loss:.4f} | 测试Loss: {avg_test_loss:.4f}")

# 最后一轮记录
history['epoch'].append(epochs - 1)
history['train_loss'].append(avg_train_loss)
history['test_loss'].append(avg_test_loss)

print(f"\n最终 | 训练Loss: {avg_train_loss:.4f} | 测试Loss: {avg_test_loss:.4f}")

# ============================================================
# Step 6: 可视化 + 保存模型
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 生成可视化图表...")
print("=" * 60)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- 图1: Loss 下降曲线 ---
ax1 = axes[0, 0]
ax1.plot(history['epoch'], history['train_loss'], 'b-o', markersize=4, label='训练集 Loss')
ax1.plot(history['epoch'], history['test_loss'], 'r-s', markersize=4, label='测试集 Loss')
ax1.set_xlabel('训练轮数 (Epoch)')
ax1.set_ylabel('MSE Loss')
ax1.set_title('图1: Loss 下降曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 图2: 预测 vs 真实散点图 ---
ax2 = axes[0, 1]
oce_names = ['Openness(O)', 'Conscient(C)', 'Extrover(E)', 'Agreeable(A)', 'Neurotic(N)']
oce_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        seqs, labels, lengths = batch
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        pred = model(seqs, lengths)
        all_preds.append(pred)
        all_labels.append(labels)

all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

for dim in range(5):
    ax2.scatter(all_labels[:50, dim], all_preds[:50, dim],
                alpha=0.5, color=oce_colors[dim], label=oce_names[dim], s=15)
ax2.plot([1, 5], [1, 5], 'k--', alpha=0.5, label='完美预测线')
ax2.set_xlabel('真实 OCEAN')
ax2.set_ylabel('预测 OCEAN')
ax2.set_title('图2: 最终预测 vs 真实标签 (测试集)')
ax2.legend(fontsize=7, loc='best')
ax2.grid(True, alpha=0.3)

# --- 图3: 各维度 MAE 柱状图 ---
ax3 = axes[1, 0]
mae_per_dim = np.abs(all_preds - all_labels).mean(axis=0)
ax3.bar(oce_names, mae_per_dim, color=oce_colors, alpha=0.7)
ax3.set_ylabel('MAE (平均绝对误差)')
ax3.set_title('图3: 各维度 MAE')
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(mae_per_dim):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

# --- 图4: 训练报告 ---
ax4 = axes[1, 1]
ax4.axis('off')

final_train_loss = history['train_loss'][-1]
final_test_loss  = history['test_loss'][-1]

text_str = (
    "================ 训练报告 ================\n\n"
    f"数据集: PSG (PersuasionForGood)\n"
    f"训练样本: {len(train_dataset)} 条\n"
    f"测试样本: {len(test_dataset)} 条\n"
    f"训练轮数: {epochs} epochs\n"
    f"Batch size: {BATCH_SIZE}\n\n"
    f"模型结构:\n"
    f"  BERT (冻结) → 逐句 [CLS](768)\n"
    f"  → Transformer Encoder(2层, 8头)\n"
    f"  → Mean Pooling\n"
    f"  → FC(768→256→128→5)\n\n"
    f"可训练参数: {total_params:,}\n\n"
    f"--- 损失变化 ---\n"
    f"最终训练 Loss: {final_train_loss:.4f}\n"
    f"最终测试 Loss: {final_test_loss:.4f}\n"
    f"MAE (各维平均): {mae_per_dim.mean():.4f}\n\n"
    f"--- 与旧版对比 (拼接+FC, 200轮) ---\n"
    f"旧版测试 MSE: 0.63\n"
    f"新版测试 MSE: {final_test_loss:.4f}"
)
ax4.text(0.05, 0.95, text_str, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(r"D:\VScode\A Try to Reproduce on Paper 'Counterfactual Reasoning'\dppr_training_result.png",
            dpi=150, bbox_inches='tight')
plt.show()

# 保存模型
torch.save(model.state_dict(), r"D:\VScode\A Try to Reproduce on Paper 'Counterfactual Reasoning'\dppr_model_weights.pth")
print("\n模型权重已保存到: dppr_model_weights.pth")
print("图表已保存到: dppr_training_result.png")
print("\n训练完成！")
print("=" * 60)
