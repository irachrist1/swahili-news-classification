"""Word-level neural classifiers.

Every model is called as model(ids, lengths) and returns logits of shape (batch, classes).
To add a model, write the class and register it in MODELS.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sequence_data import PAD_ID


def make_embedding(vocab_size, embed_dim, embeddings=None, freeze=False):
    """Embedding layer, optionally started from pretrained vectors (e.g. fastText)."""
    layer = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
    if embeddings is not None:
        layer.weight.data.copy_(torch.as_tensor(embeddings))
        layer.weight.requires_grad = not freeze
    return layer


def length_mask(lengths, max_len):
    """True for real tokens, False for padding."""
    positions = torch.arange(max_len, device=lengths.device)
    return positions[None, :] < lengths[:, None]


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM whose hidden states are pooled with additive attention.

    pooling="max" or "mean" replaces attention, for the ablation.
    The attention weights of the last batch are kept in last_attention for inspection.
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        embeddings=None,
        embed_dim=300,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
        freeze_embeddings=False,
        pooling="attention",
    ):
        super().__init__()
        self.pooling = pooling
        self.embedding = make_embedding(vocab_size, embed_dim, embeddings, freeze_embeddings)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)
        self.last_attention = None

    def forward(self, ids, lengths):
        embedded = self.embed_dropout(self.embedding(ids))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=ids.size(1))
        mask = length_mask(lengths.to(ids.device), ids.size(1))

        if self.pooling == "attention":
            scores = self.attention(outputs).squeeze(-1).masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = (weights.unsqueeze(-1) * outputs).sum(dim=1)
            self.last_attention = weights.detach()
        elif self.pooling == "max":
            pooled = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        else:
            summed = (outputs * mask.unsqueeze(-1)).sum(dim=1)
            pooled = summed / lengths.to(ids.device).unsqueeze(-1)

        return self.classifier(self.dropout(pooled))


class TextCNN(nn.Module):
    """Kim-style convolutional classifier: parallel n-gram convolutions, max-over-time pooling.

    Each kernel size looks at windows of that many consecutive words; the strongest response of each
    filter anywhere in the article is kept, so the model finds topic phrases wherever they appear.
    """

    def __init__(
        self,
        vocab_size,
        num_classes,
        embeddings=None,
        embed_dim=300,
        num_filters=100,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.embedding = make_embedding(vocab_size, embed_dim, embeddings, freeze_embeddings)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in self.kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, ids, lengths):
        longest = max(self.kernel_sizes)
        if ids.size(1) < longest:
            ids = nn.functional.pad(ids, (0, longest - ids.size(1)), value=PAD_ID)
        embedded = self.embedding(ids).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


MODELS = {
    "bilstm": BiLSTMAttention,
    "textcnn": TextCNN,
}


def build_model(name, vocab_size, num_classes, embeddings=None, **options):
    return MODELS[name](vocab_size, num_classes, embeddings=embeddings, **options)
