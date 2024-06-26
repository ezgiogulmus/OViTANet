import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp_model import MB_MLP
from models.model_utils import SimpleEncoder, SimpleFusion, Transformer, initialize_weights


class GMCAT(nn.Module):
    def __init__(
        self, nb_tabular_data=0, mm_fusion="concat", 
        mm_fusion_type="late", n_classes=4, path_input_dim=768, 
        dim=64, depth=5, mha_heads=4, 
        mlp_dim=64, pool = 'cls', dim_head = 16, 
        drop_out = 0.2, **kwargs
        ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, path_input_dim))
        self.pool = pool

        self.mm_fusion = mm_fusion if nb_tabular_data != 0 else None
        self.mm_fusion_type = mm_fusion_type if nb_tabular_data != 0 else None
        if self.mm_fusion == "crossatt":
            assert self.mm_fusion_type in ["mid", "ms"]
        
        self.patch_encoding = SimpleEncoder(path_input_dim, dim)
        if nb_tabular_data != 0:
            if isinstance(nb_tabular_data, list):
                self.tabular_encoding = MB_MLP(nb_tabular_data, target_features=dim, feat_extractor=True, mm_fusion="concat" if mm_fusion != "crossatt" else "bilinear", **kwargs)
            else:
                self.tabular_encoding = SimpleEncoder(nb_tabular_data, dim, drop_out)
        
        self.fuser = SimpleFusion(self.mm_fusion, dim) if self.mm_fusion != "crossatt" else None
        
        self.transformer = Transformer(
            dim, depth, mha_heads, dim_head, mlp_dim, drop_out, 
            mm_fusion=mm_fusion if self.mm_fusion_type in ["mid", "ms"] else None, 
            hierarchical=True if self.mm_fusion_type=="ms" else False,
            fuser=self.fuser
        )
        if self.mm_fusion == "concat" and self.mm_fusion_type == "late":
            dim *= 2
        self.classifier = nn.Linear(dim, n_classes)
        
        initialize_weights(self)


    def forward(self, **kwargs):
        # Add class token
        x = kwargs['x_path'].unsqueeze(0)
        cls_tokens = self.cls_token.expand(1, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Encode inputs
        x = self.patch_encoding(x)
        if kwargs["x_tabular"] is not None:
            tab_feats = self.tabular_encoding(kwargs["x_tabular"]).unsqueeze(0)
        
        # Early fusion
        if self.mm_fusion_type == "early":
            x = self.fuser([x, tab_feats])
        
        x = self.transformer(x, tab_feats, return_attn=kwargs["return_feats"])
        if isinstance(x, tuple):
            x, attn_weights = x
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # Late fusion
        if self.mm_fusion_type == "late":
            x = self.fuser([x, tab_feats.squeeze(1)])
        
        output = self.classifier(x)
        if kwargs["return_feats"]:
            return output, x, attn_weights
        
        return output
    
if __name__ == "__main__":
    import numpy as np
            
    np.random.seed(0)
    a = torch.FloatTensor(np.random.randn(2, 768))
    b = torch.FloatTensor(np.random.randn(1, 10))
    c = torch.FloatTensor(np.random.randn(1, 20))
    torch.random.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    print(a[0, 0], b[0, 0])
    for i in ["mid", "ms"]:
        for j in ["adaptive", "multiply", "concat", "bilinear", "crossatt"]:
            model = GMCAT(nb_tabular_data=[10, 20], mm_fusion=j, 
                mm_fusion_type=i, n_classes=4, path_input_dim=768, 
                dim=64, depth=5, mha_heads=4, 
                mlp_dim=64, pool = 'cls', dim_head = 16, 
                drop_out = 0.0
                )
            out = model(x_path=a, x_tabular=[b, c], return_feats=False)
            print(i, j, out)
            print(model)
            break
            
    # for i in ["early", "late"]:
    #     for j in ["adaptive", "multiply", "concat", "bilinear"]:
    #         model = GMCAT(nb_tabular_data=[10, 20], mm_fusion=j, 
    #             mm_fusion_type=i, n_classes=4, path_input_dim=768, 
    #             dim=64, depth=5, mha_heads=4, 
    #             mlp_dim=64, pool = 'cls', dim_head = 16, 
    #             drop_out = 0.0
    #             )
    #         out = model(x_path=a, x_tabular=[b, c], return_feats=False)
    #         print(i, j, out)
            
