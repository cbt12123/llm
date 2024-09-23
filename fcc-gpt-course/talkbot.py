import torch
import torch.nn as nn
from torch.nn import functional as F

# 设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 超参数设置
block_size = 256
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

# GPTLanguageModel 模型定义
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = x.transpose(0, 1)
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(device)
        x = self.transformer_encoder(x, mask=mask)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 加载模型
def load_model_and_vocab(model_path):
    checkpoint = torch.load(model_path)
    model = GPTLanguageModel(vocab_size=checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    string_to_int = checkpoint['string_to_int']
    int_to_string = checkpoint['int_to_string']
    return model, string_to_int, int_to_string

# 编码和解码函数
def encode(text, string_to_int):
    return [string_to_int.get(c, 0) for c in text]

def decode(indices, int_to_string):
    return ''.join([int_to_string.get(i, '') for i in indices])

# 加载模型和词汇表
model, string_to_int, int_to_string = load_model_and_vocab('gpt_model.pth')

# 对话交互
while True:
    prompt = input("Prompt: ")
    context = torch.tensor(encode(prompt, string_to_int), dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=150)
    output = decode(generated[0].tolist(), int_to_string)
    print(f'Completion: {output}')
