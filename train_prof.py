import torch
from types import SimpleNamespace
import torch.nn as nn
import time

class FakeBERT(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, words, masks): return torch.randn(words.size(0), words.size(1), 768)

args = SimpleNamespace(device=torch.device('cpu'), tokenizer_shape=768, lam_span_repr=512, lam_dropout=0.1, lam_attention_dropout=0.3)

def fake_batch(n_sent=5, max_len=40, max_char=16):
    words = torch.randint(100, 9999, (n_sent, max_len))
    text_mask = torch.ones(n_sent, max_len, dtype=torch.long)
    char_ids = torch.randint(0, 262, (n_sent, max_len, max_char))
    em, et = [], []
    for i in range(n_sent):
        nsp = 5; m, t = [], []
        for k in range(nsp):
            l = torch.randint(1, max_len - 8, (1,)).item()
            r = torch.randint(l + 2, min(l + 8, max_len), (1,)).item()
            m.append([l, r]); t.append(k % 3)
        em.append(torch.tensor(m)); et.append(torch.tensor(t))
    return {'word': words, 'text_mask': text_mask, 'entity_masks': em, 'entity_types': et,
            'sentence_num': [torch.tensor([5])] * n_sent, 'char_ids': char_ids}

N = 300
for mn in ['pacl', 'pacl_v3']:
    if mn == 'pacl': from models.pacl import pacl as M
    else: from models.pacl_v3 import pacl_v3 as M
    m = M(args, FakeBERT())
    s, q = fake_batch(), fake_batch()
    m.train()
    for _ in range(5):
        l, lg, _ = m(s, q); l.backward()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()
    for _ in range(N):
        m.zero_grad()
        l, lg, pred = m(s, q)
        l.backward()
    dt = (time.time() - t0) / N
    print(f'{mn:10s}: fwd+bwd = {dt*1000:.0f}ms/step')

    m.eval()
    with torch.no_grad():
        t1 = time.time()
        for _ in range(N):
            l2, lg2, pr = m(s, q)
        dt_fwd = (time.time() - t1) / N
    print(f'{mn:10s}: fwd_only = {dt_fwd*1000:.0f}ms/step  (no_grad)')
