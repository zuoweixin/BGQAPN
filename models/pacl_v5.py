from utils.model_framework import FewShotNERModel
import torch
from torch import nn
from torch.nn import functional as F


class BoundaryGuidedSpanAttention(nn.Module):
    
    def __init__(self, token_dim, context_dim=None):
        super().__init__()
        self.token_dim = token_dim
        context_dim = context_dim or token_dim
        self.bd_proj = nn.Linear(token_dim * 2, token_dim)
        self.in_proj = nn.Linear(token_dim, token_dim)
        self.ctx_proj = nn.Linear(context_dim, token_dim)
        self.fuse = nn.Sequential(nn.Linear(token_dim * 3, token_dim), nn.Tanh())
        self.gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, span_tokens, span_mask, wl, wr, sent_ctx):
        bd = torch.cat([wl, wr], dim=-1)
        q = torch.tanh(self.bd_proj(bd))
        k = self.in_proj(span_tokens)
        ctx_rep = self.ctx_proj(sent_ctx)
        scores = (q.unsqueeze(1) * k).sum(dim=-1)
        scores = scores.masked_fill(~span_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        h_in = (span_tokens * weights.unsqueeze(-1)).sum(dim=1)
        h_bd = (wl + wr) / 2
        h_fused = self.fuse(torch.cat([h_bd, h_in, ctx_rep], dim=-1))
        g = torch.sigmoid(self.gate)
        return g * h_bd + (1 - g) * h_fused


class pacl_v5(FewShotNERModel):
    """
    pacl_v5：span表示 + v5的EM原型/对比损失 
    """
    def __init__(self, args, word_encoder):
        FewShotNERModel.__init__(self, args, word_encoder)
        self.drop = nn.Dropout()
        self.args = args
        self.token_dim = args.tokenizer_shape

        # v4：BG-Attn + LASC
        self.span_attn = BoundaryGuidedSpanAttention(self.token_dim)

        # v4关键：fusion_linear回归"端点concat + BG-Attn"双通道
        # 原版PAC L: Linear(768*2 → 768 → 512)
        # v3退化:    Linear(768 → 768 → 512)        (丢端点raw→ACE05↓)
        # v4/v5:     Linear(768*2 + 768 → 768 → 512)  (wl,wr concat + BG-Attn)
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.token_dim * 2 + self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(args.lam_dropout),
            nn.Linear(self.token_dim, args.lam_span_repr),
        )

        self.attentioner = nn.MultiheadAttention(args.lam_span_repr, 1, batch_first=True)
        self.layernorm = nn.LayerNorm(args.lam_span_repr, elementwise_affine=False)

        # 校准参数
        self.O_type_theta = nn.Parameter(torch.tensor([0.3]))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.theta_gamma = nn.Parameter(torch.tensor([-0.2]))
        self.scale_lambda = 0.1
        self.diversity_weight = nn.Parameter(torch.tensor([1e-3]))

        # v5新增：质量评估网络
        self.quality_scorer = nn.Sequential(
            nn.Linear(args.lam_span_repr, args.lam_span_repr // 2),
            nn.Tanh(),
            nn.Linear(args.lam_span_repr // 2, 1)
        )
        self.temperature = nn.Parameter(torch.tensor([0.07]))
        self.CELoss = nn.CrossEntropyLoss()

        # 开关
        self.use_sqr = hasattr(args, 'use_sqr') and bool(getattr(args, 'use_sqr', True))
        self.use_div = hasattr(args, 'use_div') and bool(getattr(args, 'use_div', True))
        self.use_calibrate = hasattr(args, 'use_calibrate') and bool(getattr(args, 'use_calibrate', True))
        self.use_lasc = hasattr(args, 'use_lasc') and bool(getattr(args, 'use_lasc', True))

     # 消融实验开关
        self.no_srm = hasattr(args, 'no_srm') and bool(getattr(args, 'no_srm', False))
        self.no_pcm = hasattr(args, 'no_pcm') and bool(getattr(args, 'no_pcm', False))
        self.no_icm = hasattr(args, 'no_icm') and bool(getattr(args, 'no_icm', False))

    def get_batch_embedding(self, batch_data):
        """分句feed-forward，峰值显存≈单句max(span数)而不是总span数"""
        word_emb = self.word_encoder(batch_data['word'], batch_data['text_mask'])

        all_span_repr = []
        all_bd_concat = []
        for emb, entity_mask in zip(word_emb, batch_data['entity_masks']):
            n_spans = entity_mask.size(0)
            if n_spans == 0:
                continue

            span_left = entity_mask[:, 0].clamp(0, emb.size(0) - 1)
            span_right_idx = (entity_mask[:, 1] - 1).clamp(0, emb.size(0) - 1)
            wl = emb.index_select(0, span_left)
            wr = emb.index_select(0, span_right_idx)
            all_bd_concat.append(torch.cat([wl, wr], dim=-1))

            span_len = (entity_mask[:, 1] - entity_mask[:, 0]).clamp(1, emb.size(0))
            max_len = int(span_len.max().item())
            t_idx = torch.arange(max_len, device=emb.device)
            idx = (span_left.unsqueeze(1) + t_idx.unsqueeze(0)).clamp(0, emb.size(0) - 1)
            span_mask = t_idx.unsqueeze(0) < span_len.unsqueeze(1)
            span_tokens = emb.index_select(0, idx.flatten()).reshape(n_spans, max_len, emb.size(-1))

            if self.use_lasc:
                sent_ctx = emb.mean(dim=0, keepdim=True).expand(n_spans, -1)
            else:
                sent_ctx = torch.zeros(n_spans, emb.size(-1), device=emb.device)
            # SRM模块未消融之前
            #h_span = self.span_attn(span_tokens, span_mask, wl, wr, sent_ctx)

            #--------------------------------------SRM消融开关--------------------------------------------
            if not self.no_srm:
                h_span = self.span_attn(span_tokens, span_mask, wl, wr, sent_ctx)
            else:
                h_span = (wl + wr) / 2
            all_span_repr.append(h_span)

            # 爆显存修复：及时释放该句的中间张量
            del span_tokens, span_mask, span_left, span_right_idx, wl, wr, span_len

        if not all_span_repr:
            return torch.zeros(0, self.token_dim, device=word_emb.device)

        h_span_all = torch.cat(all_span_repr, 0)
        bd_all = torch.cat(all_bd_concat, 0)
        total = torch.cat([bd_all, h_span_all], dim=-1)
        del bd_all, h_span_all
        out = self.fusion_linear(total)
        del total
        return out

    def get_proto(self, embedding, tag):
        """v5创新：质量评估 + EM迭代去噪原型"""
        proto, proto_label = [], []
        total_label = list(set(tag.tolist()))
        total_label.sort()
        for label in total_label:
            words = embedding[tag == label]
            if words.size(0) <= 1:
                proto.append(words.squeeze(0))
                proto_label.append(label)
                continue
            p_mean = torch.mean(words, dim=0)
            
            #--------------------------------------PCM消融开关--------------------------------------------
            if self.no_pcm:
                proto.append(p_mean)
                proto_label.append(label)
                continue
            quality_scores = self.quality_scorer(words).squeeze(-1)
            quality_w = torch.sigmoid(quality_scores)
            quality_w = quality_w / (quality_w.sum() + 1e-6)
            p = (words * quality_w.unsqueeze(-1)).sum(dim=0)

            for _ in range(3):
                dists = torch.norm(words - p.unsqueeze(0), dim=-1)
                g_w = torch.exp(-dists / (dists.std() + 1e-6))
                combined_w = g_w * quality_w
                combined_w = combined_w / (combined_w.sum() + 1e-6)
                p = (words * combined_w.unsqueeze(-1)).sum(dim=0)
                del dists, g_w, combined_w

            proto.append(p)
            proto_label.append(label)

        proto = torch.stack(proto) if proto else torch.zeros(0, embedding.size(-1), device=embedding.device)
        proto_label = torch.tensor(proto_label, device=embedding.device) if proto_label else torch.zeros(0, dtype=torch.long, device=embedding.device)
        return proto, proto_label

    def conditional_diversity_loss(self, proto_nonO, alpha=0.7):
        if not self.use_div or proto_nonO.size(0) < 2:
            return torch.tensor(0.0, device=proto_nonO.device)
        proto_n = F.normalize(proto_nonO, dim=-1)
        sim = proto_n @ proto_n.t()
        n = sim.size(0)
        mask = 1 - torch.eye(n, device=sim.device)
        off_diag = sim * mask
        high_sim = F.relu(off_diag - alpha) ** 2
        return high_sim.sum() / (n * (n - 1))

    def calibrate_theta(self, proto_nonO):
        if not self.use_calibrate or proto_nonO.size(0) < 2:
            return torch.sigmoid(torch.tensor(0.5, device=proto_nonO.device))
        proto_n = F.normalize(proto_nonO, dim=-1)
        sim = proto_n @ proto_n.t()
        n = sim.size(0)
        triu_idx = torch.triu_indices(n, n, offset=1, device=sim.device)
        S_min = sim[triu_idx].min().item()
        return torch.sigmoid(self.O_type_theta) + torch.sigmoid(self.theta_gamma) * S_min

    def calibrate_scale(self, proto_nonO):
        if not self.use_calibrate or proto_nonO.size(0) < 2:
            return torch.exp(self.log_scale)
        proto_n = F.normalize(proto_nonO, dim=-1)
        sim = proto_n @ proto_n.t()
        n = sim.size(0)
        triu_idx = torch.triu_indices(n, n, offset=1, device=sim.device)
        S_min = sim[triu_idx].min().item()
        return torch.exp(self.log_scale) + self.scale_lambda * (1 - S_min)



    def MHA_encode(self, proto, hidden_states, drop=True):
        residual = hidden_states
        atten = self.attentioner(hidden_states, proto, proto)[0]
        if drop:
            atten = nn.functional.dropout(atten, p=self.args.lam_attention_dropout, training=self.training)
        hidden_states = residual + atten
        hidden_states = self.layernorm(hidden_states)
        return hidden_states.squeeze(0)

    def MHA_encode_bidirectional(self, proto, hidden_states, drop=True):
        """双向互注意力：query→proto + proto←query，让原型也自适应query分布"""
        # 方向1：query 看 proto（和原始一致）
        residual = hidden_states
        atten = self.attentioner(hidden_states, proto, proto)[0]
        if drop:
            atten = nn.functional.dropout(atten, p=self.args.lam_attention_dropout, training=self.training)
        hidden_states = residual + atten
        query_out = self.layernorm(hidden_states).squeeze(0)

        # 方向2：proto 看 query（新增，让原型自适应query）
        proto_residual = proto
        proto_atten = self.attentioner(proto, hidden_states, hidden_states)[0]
        if drop:
            proto_atten = nn.functional.dropout(proto_atten, p=self.args.lam_attention_dropout, training=self.training)
        proto_out = proto_residual + proto_atten
        proto_out = self.layernorm(proto_out).squeeze(0)
        return query_out, proto_out

    def forward(self, support, query):
        spt = self.get_batch_embedding(support)
        spt_label = torch.cat(support['entity_types'], 0)
        query_emb = self.get_batch_embedding(query)

        support_proto, proto_label = self.get_proto(self.drop(spt), spt_label)
        if support_proto.size(0) == 0:
            return self.training, torch.zeros(1, self.args.lam_span_repr, device=spt.device), torch.zeros(1, dtype=torch.long, device=spt.device)

        nonO_proto = support_proto[1:]
        if nonO_proto.size(0) == 0:
            logits = torch.zeros(query_emb.size(0), 1, device=query_emb.device)
            return None, logits, torch.zeros(query_emb.size(0), dtype=proto_label.dtype, device=query_emb.device)

    
        # -------------------------------ICM消融开关--------------------------------------------
        if not self.no_icm:
            query_emb_2, nonO_proto = self.MHA_encode_bidirectional(
                nonO_proto.unsqueeze(0), query_emb.unsqueeze(0)
            )
        else:
            query_emb_2 = query_emb  # 消融时归一化，避免尺度爆炸

        Q_1 = F.normalize(query_emb, dim=-1)
        S = F.normalize(nonO_proto, dim=-1)
        dist = (query_emb_2[:, None, :] * S).sum(-1)
        dist_O = (Q_1 * query_emb_2).sum(-1)
        if not self.no_icm:
            theta = self.calibrate_theta(nonO_proto)
            scale = self.calibrate_scale(nonO_proto)
        else:
            # 消融 ICM：保留基础 theta 学习，移除自适应和 scale
            theta = torch.sigmoid(self.O_type_theta)
            scale = torch.exp(self.log_scale)
        dist = dist * scale
        dist_O = dist_O * theta * scale
        
        logits = torch.cat([dist_O.unsqueeze(1), dist], dim=1)
        pred = self.inference_label(logits, proto_label)

        del Q_1, S, dist, dist_O

        if not self.training:
            return None, logits, pred

        qry_label = torch.cat(query['entity_types'], 0)
        if proto_label.max().item() != len(set(proto_label.tolist())) - 1:
            label2idx = {i: idx for idx, i in enumerate(proto_label.tolist())}
            qry_label = torch.tensor([label2idx[i] for i in qry_label.tolist()]).to(qry_label.device)

        loss_q = self.loss(logits, qry_label)
        div_loss = self.conditional_diversity_loss(nonO_proto)
        
        total_loss = loss_q + torch.sigmoid(self.diversity_weight) * div_loss 
        
        # 每500步清一次CUDA碎片（爆显存修复）
        if self.training and torch.is_grad_enabled():
            if hasattr(self, '_step_cnt'):
                self._step_cnt += 1
            else:
                self._step_cnt = 1
            if self._step_cnt % 1000 == 0:
                torch.cuda.empty_cache()

        return total_loss, logits, pred

    def loss(self, logits, label):
        label = label.view(-1).to(logits.device)
        dim = logits.shape[1]
        loss_weights = F.one_hot(label, dim).view(-1, dim)
        loss_mask = 1 - loss_weights
        loss_pos = (torch.exp(-logits) * loss_weights).sum(dim=-1)
        loss_neg = (torch.exp(logits) * loss_mask).sum(dim=-1)
        return torch.log(1 + loss_pos * loss_neg).mean()

    def inference_label(self, logits, proto_label):
        _, pred_direct = torch.max(logits, 1)
        return proto_label[pred_direct]

    def test(self, query, spt_total, require_emb=False):
        spt = spt_total[0]
        spt_label = spt_total[1]
        query_emb = self.get_batch_embedding(query)

        support_proto, proto_label = self.get_proto(self.drop(spt), spt_label)
        if support_proto.size(0) == 0:
            logits = torch.zeros(query_emb.size(0), 1, device=query_emb.device)
            pred = torch.zeros(query_emb.size(0), dtype=proto_label.dtype, device=query_emb.device)
            if require_emb:
                return logits, pred, query_emb, query_emb
            return logits, pred

        nonO_proto = support_proto[1:]
        if nonO_proto.size(0) == 0:
            logits = torch.zeros(query_emb.size(0), 1, device=query_emb.device)
            pred = torch.zeros(query_emb.size(0), dtype=proto_label.dtype, device=query_emb.device)
            if require_emb:
                return logits, pred, query_emb, query_emb
            return logits, pred
     
         # === 单向互注意力（仅测试时）===
        if not self.no_icm:
            query_emb_2 = self.MHA_encode(nonO_proto.unsqueeze(0), query_emb.unsqueeze(0))
        else:
            query_emb_2 = query_emb  # 同上
        Q_1 = F.normalize(query_emb, dim=-1)
        S = F.normalize(nonO_proto, dim=-1)
        dist = (query_emb_2[:, None, :] * S).sum(-1)
        dist_O = (Q_1 * query_emb_2).sum(-1)
        if not self.no_icm:
            theta = self.calibrate_theta(nonO_proto)
            scale = self.calibrate_scale(nonO_proto)
        else:
            theta = torch.sigmoid(self.O_type_theta)
            scale = torch.exp(self.log_scale)

        dist = dist * scale
        dist_O = dist_O * theta * scale
        logits = torch.cat([dist_O.unsqueeze(1), dist], dim=1)

        pred = self.inference_label(logits, proto_label)
        if require_emb:
            return logits, pred, query_emb, query_emb_2
        return logits, pred
