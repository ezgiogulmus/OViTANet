import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import SimpleFusion

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
    def __init__(self, nb_tabular_data, mlp_type="small", activation="relu", drop_out=.25, target_features=50, n_classes=1, mlp_skip=True, feat_extractor=False, batch_norm=False, **kwargs):
        super(MLP, self).__init__()

        self.skip = mlp_skip
        self.feat_extractor = feat_extractor
        
        self.activations = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(),
            'gelu': nn.GELU()
        }
        assert activation in self.activations.keys(), "Unknown activation function."
        
        if mlp_type == "small":
            hidden_dims = [nb_tabular_data, 256, 128, target_features]
            self.skip = False
        else:
            hidden_dims = [nb_tabular_data, 4096, 2048, 1024, 512, 256, 128, target_features]
                    
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


class MB_MLP(nn.Module):
    def __init__(self, nb_tabular_data, target_features=50, n_classes=1, feat_extractor=False, mm_fusion="concat", **kwargs):
        super(MB_MLP, self).__init__()
        self.feat_extractor = feat_extractor
        self.encoders = nn.ModuleList([MLP(nb_tabular_data[i], feat_extractor=True, target_features=target_features, **kwargs) for i in range(len(nb_tabular_data))])
        self.fuser = None
        if mm_fusion == "bilinear":
            self.fuser = SimpleFusion(mm_fusion, target_features, nb_of_vectors=len(nb_tabular_data))
            
        if not feat_extractor:
            self.classifier = nn.Linear(target_features, n_classes)

    def forward(self, omics_list):
        assert len(omics_list) == len(self.encoders), "Something's wrong!"
        encoded_omics = [self.encoders[i](omics_list[i]) for i in range(len(omics_list))]
        fused_omics = self.fuser(encoded_omics) if self.fuser is not None else torch.cat(encoded_omics, dim=0)
        
        if self.feat_extractor:
            return fused_omics
        return self.classifier(fused_omics)
        

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # model = MLP(1000, mlp_type="big", activation="relu", drop_out=.25, target_features=50, n_classes=1, skip=True, feat_extractor=False, batch_norm=False)
    import numpy as np
            
    np.random.seed(0)
    dim1, dim2, dim3 = 100, 200, 400
    a = torch.FloatTensor(np.random.randn(1, dim1))
    b = torch.FloatTensor(np.random.randn(1, dim2))
    c = torch.FloatTensor(np.random.randn(1, dim3))
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    print(a[0, 0], b[0, 0], c[0, 0])
    for fusion in ["concat", "bilinear"]:
        model = MB_MLP([dim1, dim2, dim3], mm_fusion=fusion)
        out = model([a, b, c])
        print(fusion, out)