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
    def __init__(self, input_dim, mlp_depth=5, mlp_type="small", activation="relu", drop_out=.25, target_features=50, n_classes=1, skip=True, feat_extractor=False, batch_norm=False):
        super(MLP, self).__init__()

        self.skip = skip
        self.feat_extractor = feat_extractor
        
        self.activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }
        assert activation in self.activations.keys(), "Unknown activation function."

        # Define sizes based on model type
        if input_dim < 100 or mlp_type=="tiny":
            self.skip = False
            self.blocks = nn.ModuleList([MLPBlock(input_dim, 256, self.activations[activation], drop_out, bn=batch_norm)])
            sizes = [256]
        else:
            start_dim = 1024 if mlp_type == "small" else 2048
            end_dim = 128 if mlp_type == "small" else 256
            sizes = torch.logspace(torch.log10(torch.tensor(start_dim)), 
                                        torch.log10(torch.tensor(end_dim)), 
                                        steps=mlp_depth-1)
            sizes = torch.round(sizes).int().tolist()
            sizes.insert(0, input_dim)
            i = 1
            
            self.blocks = nn.ModuleList([MLPBlock(sizes[i-1], sizes[i], self.activations[activation], drop_out, bn=batch_norm) for i in range(1, mlp_depth)])
        if self.skip:
            skip_indices = [0] + [i for i in range(1, len(sizes), 2)]
            skip_sizes = [sizes[i] for i in skip_indices]
            # skip_sizes = [sizes[i] for i in range(0, len(sizes), 2)]
            # print(sizes, skip_sizes)
            self.skip_blocks = nn.ModuleList([MLPBlock(skip_sizes[i], skip_sizes[i+1], self.activations[activation], drop_out, bn=batch_norm) for i in range(len(skip_sizes)-1)])
            
        self.to_feats = nn.Linear(sizes[-1], target_features)
        
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
                # print("Adding skip: ", x.shape, skip_x.shape)
                x = x + skip_x
                skip_idx += 1

        x = self.to_feats(x)
        if self.feat_extractor:
            return x
        x = self.final(x)
        return x

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

