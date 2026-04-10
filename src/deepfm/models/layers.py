from __future__ import annotations

import torch
from torch import nn


class FeaturesEmbedding(nn.Module):
    def __init__(self, cardinalities: list[int], embed_dim: int) -> None:
        super().__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(cardinality, embed_dim) for cardinality in cardinalities]
        )
        for layer in self.embedding_layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [layer(x[:, index]) for index, layer in enumerate(self.embedding_layers)]
        return torch.stack(embeddings, dim=1)


class FeaturesLinear(nn.Module):
    def __init__(self, cardinalities: list[int], dense_feature_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.sparse_linear = nn.ModuleList([nn.Embedding(cardinality, 1) for cardinality in cardinalities])
        self.dense_linear = nn.Linear(dense_feature_dim, 1) if dense_feature_dim > 0 else None

        for layer in self.sparse_linear:
            nn.init.zeros_(layer.weight)
        if self.dense_linear is not None:
            nn.init.xavier_uniform_(self.dense_linear.weight)
            nn.init.zeros_(self.dense_linear.bias)

    def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor) -> torch.Tensor:
        outputs = self.bias.expand(dense_x.size(0), 1)
        if self.dense_linear is not None and dense_x.numel() > 0:
            outputs = outputs + self.dense_linear(dense_x)
        if sparse_x.numel() > 0:
            sparse_terms = [layer(sparse_x[:, index]) for index, layer in enumerate(self.sparse_linear)]
            outputs = outputs + torch.stack(sparse_terms, dim=1).sum(dim=1)
        return outputs


class FactorizationMachine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summed = x.sum(dim=1)
        summed_square = summed.pow(2)
        squared_sum = x.pow(2).sum(dim=1)
        interactions = 0.5 * (summed_square - squared_sum).sum(dim=1, keepdim=True)
        return interactions


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
