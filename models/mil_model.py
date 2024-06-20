import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_fc_mc(nn.Module):
    def __init__(self, drop_out = 0., n_classes = 2, path_input_dim=1024, **kwargs):
        super().__init__()
        assert n_classes > 2
        
        fc = [nn.Linear(path_input_dim, 512), nn.ReLU(), nn.Dropout(drop_out)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(512, n_classes)
        self.n_classes = n_classes
    
    def forward(self, **kwargs):       
        h = self.fc(kwargs["x_path"])
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]
        
        if kwargs["return_feats"]:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            return top_instance, top_features, y_probs
        return top_instance