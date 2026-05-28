import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN: 多尺寸卷积核 (3/4/5-gram) 拼接"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_filters: int = 100,
        filter_sizes: tuple = (3, 4, 5),
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

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        emb = emb.unsqueeze(1)           # (batch, 1, seq_len, embed_dim)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))           # (batch, num_filters, seq_len - fs + 1, 1)
            c = c.squeeze(3)                # (batch, num_filters, L_out)
            c = F.max_pool1d(c, c.size(2))  # (batch, num_filters, 1)
            pooled.append(c.squeeze(2))     # (batch, num_filters)

        out = torch.cat(pooled, dim=1)      # (batch, num_filters * 3)
        out = self.dropout(out)
        return self.fc(out)
