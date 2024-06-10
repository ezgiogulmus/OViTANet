import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, drop_out, bn=False):
        super(MLPBlock, self).__init__()
        self.bn = bn
        self.linear = nn.Linear(in_features, out_features)
        if bn:
            self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(drop_out)
        initialize_weights(self)

    def forward(self, x):
        x = self.linear(x)
        if self.bn:
            x = self.batch_norm(x)
        return self.dropout(self.activation_fn(x))

class MLP(nn.Module):
    def __init__(self, input_dim, mlp_type="small", activation="relu", drop_out=.25, target_features=50, n_classes=1, skip=True, feat_extractor=False, batch_norm=False):
        super(MLP, self).__init__()

        self.skip = skip
        self.feat_extractor = feat_extractor
        
        self.activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }
        assert activation in self.activations.keys(), "Unknown activation function."
        
        if mlp_type == "small":
            hidden_dims = [input_dim, 256, 128, target_features]
            self.skip = False
        # elif mlp_type == "small":
        #     hidden_dims = [input_dim, 1024, 512, 256, 128, target_features]
        else:
            hidden_dims = [input_dim, 4096, 2048, 1024, 512, 256, 128, target_features]
                    
        self.blocks = nn.Sequential(*[
            MLPBlock(hidden_dims[i], hidden_dims[i+1], self.activations[activation], drop_out, bn=batch_norm) 
            for i in range(len(hidden_dims)-1)
        ])
        
        if self.skip:
            skip_indices = [0] + [i for i in range(1, len(hidden_dims), 2)]
            skip_dims = [hidden_dims[i] for i in skip_indices]
            self.skip_blocks = nn.Sequential(*[
                MLPBlock(skip_dims[i], skip_dims[i+1], self.activations[activation], drop_out, bn=batch_norm) 
                for i in range(len(skip_dims)-1)
            ])
        
        if not feat_extractor:
            self.final = nn.Linear(target_features, n_classes)

        initialize_weights(self)

    def forward(self, x):
        skip_x = x
        skip_idx = 0

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.skip and i % 2 == 0:
                skip_x = self.skip_blocks[skip_idx](skip_x)
                x = x + skip_x
                skip_idx += 1

        if self.feat_extractor:
            return x
        return self.final(x)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

