import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """加性注意力 (Bahdanau-style)"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output, mask=None):
        # gru_output: (batch, seq_len, hidden_dim * 2)
        scores = self.v(torch.tanh(self.W(gru_output)))  # (batch, seq_len, 1)

        if mask is not None:
            # mask: (batch, seq_len) -> 把 pad 位置的 score 变成 -inf
            mask = mask.unsqueeze(-1).float()
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * gru_output, dim=1)  # (batch, hidden_dim * 2)
        return context, attn_weights.squeeze(-1)


class BiGRUAttention(nn.Module):
    """BiGRU + Attention 文本分类"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        # GRU
        gru_out, _ = self.gru(emb)  # (batch, seq_len, hidden_dim * 2)

        # Attention
        context, _ = self.attention(gru_out, attention_mask)  # (batch, hidden_dim * 2)

        out = self.dropout(context)
        return self.fc(out)
