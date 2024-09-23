import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import os
import zipfile

# 超参数设置
batch_size = 8
block_size = 256
max_iters = 50000
learning_rate = 3e-4
eval_interval = 500
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

# 设备检查
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 文件路径
train_zip_path = 'datasets/ijcnlp_dailydialog/train.zip'
valid_zip_path = 'datasets/ijcnlp_dailydialog/validation.zip'
train_folder = 'datasets/ijcnlp_dailydialog/train'
valid_folder = 'datasets/ijcnlp_dailydialog/validation'

# 解压数据集
def unzip_file(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"解压 {zip_path} 到 {extract_to}")
    else:
        print(f"{extract_to} 已存在，无需解压")

# 解压训练和验证集
unzip_file(train_zip_path, train_folder)
unzip_file(valid_zip_path, valid_folder)

train_path = os.path.join(train_folder, 'train', 'dialogues_train.txt')
valid_path = os.path.join(valid_folder, 'validation', 'dialogues_validation.txt')

# 加载文本数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dialogues = f.readlines()
    return [dialogue.strip() for dialogue in dialogues]

train_data_raw = load_data(train_path)
valid_data_raw = load_data(valid_path)

# 数据预处理
def preprocess_dialogue(dialogue):
    sentences = dialogue.split('__eou__')
    return [s.strip() for s in sentences if s.strip()]

processed_train_data = [preprocess_dialogue(d) for d in train_data_raw]
processed_valid_data = [preprocess_dialogue(d) for d in valid_data_raw]

# 构建词汇表
def build_vocab(data):
    chars = set()
    for dialogue in data:
        for sentence in dialogue:
            chars.update(sentence)
    vocab = sorted(list(chars))
    vocab_size = len(vocab)
    string_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_string = {i: ch for i, ch in enumerate(vocab)}
    return string_to_int, int_to_string, vocab_size

string_to_int, int_to_string, vocab_size = build_vocab(processed_train_data)
print(f"词汇表大小: {vocab_size}")

# 编码函数
def encode_dialogue(dialogue):
    dialogue_text = ' '.join(dialogue)
    return [string_to_int.get(c, 0) for c in dialogue_text]

# 批次数据生成
def create_batches(data, batch_size, block_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batch_dialogues = data[i:i+batch_size]
        batch_encoded = []
        for dialogue in batch_dialogues:
            encoded = encode_dialogue(dialogue)
            if len(encoded) > block_size:
                encoded = encoded[:block_size]
            else:
                encoded += [0] * (block_size - len(encoded))
            batch_encoded.append(encoded)
        batch_tensor = torch.tensor(batch_encoded, dtype=torch.long).to(device)
        batches.append(batch_tensor)
    return batches

train_batches = create_batches(processed_train_data, batch_size, block_size)
valid_batches = create_batches(processed_valid_data, batch_size, block_size)

# 模型定义
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        position_embeddings = self.position_embedding_table(position_ids)  # (B,T,C)
        x = token_embeddings + position_embeddings  # (B,T,C)
        x = x.transpose(0, 1)  # Transformer expects input as (T,B,C)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(device)
        x = self.transformer_encoder(x, mask=mask)
        x = x.transpose(0, 1)  # Back to (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets is not None:
            # 使用 reshape 替换 view 以避免 stride 错误
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # 只考虑最后一个 token 的预测
            logits = logits / temperature  # 调整 temperature
            
            if top_k is not None:
                # Top-k 策略：只保留前 k 个概率最高的词
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx


# 模型初始化
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练过程
def estimate_loss():
    model.eval()
    losses = {'train': 0, 'valid': 0}
    with torch.no_grad():
        for split in ['train', 'valid']:
            data = train_batches if split == 'train' else valid_batches
            total_loss = 0
            for batch in data:
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                _, loss = model(inputs, targets)
                total_loss += loss.item()
            losses[split] = total_loss / len(data)
    model.train()
    return losses

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"步骤 {iter}: 训练集损失 {losses['train']:.4f}, 验证集损失 {losses['valid']:.4f}")

    batch = random.choice(train_batches)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    optimizer.zero_grad()
    logits, loss = model(inputs, targets)
    loss.backward()
    optimizer.step()

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'string_to_int': string_to_int,
    'int_to_string': int_to_string,
    'vocab_size': vocab_size,
    }, 'gpt_model.pth')
print("模型已保存。")
