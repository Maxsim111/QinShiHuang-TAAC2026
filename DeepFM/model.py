from __future__ import annotations

import torch
from torch import nn

try:
    from .layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MLP
except ImportError:
    from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MLP


class DeepFM(nn.Module):
    def __init__(
        self,
        dense_feature_dim: int,
        sparse_cardinalities: list[int],
        embed_dim: int,
        hidden_units: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dense_feature_dim = dense_feature_dim
        self.sparse_feature_dim = len(sparse_cardinalities)
        self.embed_dim = embed_dim

        self.linear = FeaturesLinear(sparse_cardinalities, dense_feature_dim)
        self.fm = FactorizationMachine()
        self.sparse_embedding = FeaturesEmbedding(sparse_cardinalities, embed_dim) if sparse_cardinalities else None
        self.dense_embedding = (
            nn.Parameter(torch.empty(dense_feature_dim, embed_dim)) if dense_feature_dim > 0 else None
        )
        if self.dense_embedding is not None:
            nn.init.xavier_uniform_(self.dense_embedding)

        mlp_input_dim = dense_feature_dim + self.sparse_feature_dim * embed_dim
        self.mlp = MLP(mlp_input_dim, hidden_units, dropout)

    def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor) -> torch.Tensor:
        linear_term = self.linear(dense_x, sparse_x)

        fm_parts = []
        sparse_emb = None
        if self.sparse_embedding is not None and sparse_x.numel() > 0:
            sparse_emb = self.sparse_embedding(sparse_x)
            fm_parts.append(sparse_emb)
        if self.dense_embedding is not None and dense_x.numel() > 0:
            dense_emb = dense_x.unsqueeze(-1) * self.dense_embedding.unsqueeze(0)
            fm_parts.append(dense_emb)

        fm_term = self.fm(torch.cat(fm_parts, dim=1)) if fm_parts else torch.zeros_like(linear_term)

        deep_parts = []
        if dense_x.numel() > 0:
            deep_parts.append(dense_x)
        if sparse_emb is not None:
            deep_parts.append(sparse_emb.flatten(start_dim=1))
        deep_input = torch.cat(deep_parts, dim=1) if deep_parts else torch.zeros_like(linear_term)
        deep_term = self.mlp(deep_input)
        return linear_term + fm_term + deep_term
