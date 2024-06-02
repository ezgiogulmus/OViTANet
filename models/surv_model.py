import torch
import torch.nn as nn


class AggregationModule(nn.Module):
    def __init__(self, path_input_dim):
        super(AggregationModule, self).__init__()
        self.attention_weights = nn.Linear(path_input_dim, 1)
    
    def forward(self, features):
        attn_scores = self.attention_weights(features)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_feats = features * attn_weights
        return torch.sum(weighted_feats, dim=1), attn_weights


class SurvMIL(nn.Module):
    def __init__(self, path_input_dim, n_classes, **kwargs):
        super(SurvMIL, self).__init__()
        self.pooling = AggregationModule(path_input_dim)
        self.fc = nn.Linear(path_input_dim, n_classes)
    
    def forward(self, **kwargs):
        x = torch.unsqueeze(kwargs['x_path'], 0)
        pooled_features, attn_weights = self.pooling(x)
        output = self.fc(pooled_features)
        if kwargs["return_feats"]:
            return output, pooled_features, attn_weights
        return output







