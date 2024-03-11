import torch
from torch import nn

from models.model_utils import BilinearFusion, LRBilinearFusion
from models.mlp_model import MLP

# from model_utils import BilinearFusion, LRBilinearFusion
# from mlp_model import MLP

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.tab_to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, img, tab=None, return_weights=False):
        img = self.norm(img)

        qkv = self.to_qkv(img).chunk(3, dim = -1)
        # print("QKV", qkv[0].shape)
        b, n = qkv[0].size(0), qkv[0].size(1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        if tab is not None:
            tab = self.norm(tab)
            qkv_tab = self.tab_to_qkv(tab).chunk(3, dim = -1)
            _, k_tab, v_tab = map(lambda t: t.view(b, 1, self.heads, self.dim_head).transpose(1, 2), qkv_tab)
            k = torch.cat([k, k_tab], dim=2)
            v = torch.cat([v, v_tab], dim=2)
        # print("k", k.shape, q.shape, k_tab.shape)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        dropped_attn = self.dropout(attn)
        # print("d", dots.shape, attn.shape, v.shape)
        out = torch.matmul(dropped_attn, v)
        # print(out.shape)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        
        out = self.to_out(out)
        if not return_weights:
            return out
        return out, attn#.transpose(1, 2).reshape(b, n, -1)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., mm_fusion=None):
        super().__init__()
        self.mm_fusion = mm_fusion
        if self.mm_fusion == "adaptive":
            self.tab_weights = nn.Parameter(torch.ones(dim))
            self.img_weights = nn.Parameter(torch.ones(dim))
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, tab=None):
        for i, (attn, ff) in enumerate(self.layers):
            if self.mm_fusion == "crossatt":
                
                if i == len(self.layers) - 1:
                    att_out, attn_weights = attn(x, tab=tab, return_weights=True)
                else:
                    att_out = attn(x, tab=tab)
                x = att_out + x
            else:
                if i == len(self.layers) - 1:
                    att_out, attn_weights = attn(x, return_weights=True)
                else:
                    att_out = attn(x)
                x = att_out + x
                if self.mm_fusion == "concat":
                    x = torch.cat([x, tab], dim=1)
                elif self.mm_fusion == "adaptive":
                    x = tab * self.tab_weights + self.img_weights * x
                elif self.mm_fusion == "multiply":
                    x = torch.mul(tab, x)
            x = ff(x) + x

        return self.norm(x), attn_weights


class ViT(nn.Module):
    def __init__(
        self, nb_tabular_data=0, mm_fusion="concat", 
        mm_fusion_type="late", n_classes=4, path_input_dim=768, 
        target_features=50, depth=5, mha_heads=4, 
        model_dim=None, mlp_dim=64, pool = 'cls', dim_head = 16, 
        drop_out = 0.2, emb_dropout = 0.5, **kwargs
        ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        if model_dim:
            self.model_dim = int(model_dim)
            self.fc = nn.Linear(path_input_dim, self.model_dim)
        else:
            self.model_dim = path_input_dim
            self.fc = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mm_fusion = mm_fusion if nb_tabular_data > 0 else None
        self.mm_fusion_type = mm_fusion_type if nb_tabular_data > 0 else None
        if self.mm_fusion_type == "early":
            assert self.mm_fusion in ["concat", "adaptive", "multiply"], "Fusion method is not implemented for the selected level."
        elif self.mm_fusion_type == "mid":
            assert self.mm_fusion in ["concat", "adaptive", "multiply", "crossatt"], "Fusion method is not implemented for the selected level."
        elif self.mm_fusion_type == "late":
            assert self.mm_fusion in ["concat", "adaptive", "multiply", "bilinear"], "Fusion method is not implemented for the selected level."

        target_tab_features = target_features if self.mm_fusion_type == "late" else path_input_dim

        if nb_tabular_data > 0:
            self.tabular_emb = MLP(
                nb_tabular_data, 
                drop_out=drop_out,
                target_features=target_tab_features, 
                feat_extractor=True,
                **kwargs)

        self.transformer = Transformer(self.model_dim, depth, mha_heads, dim_head, mlp_dim, drop_out, mm_fusion=None if self.mm_fusion_type != "mid" else mm_fusion)
        self.to_feats = nn.Linear(self.model_dim, target_features)

        if self.mm_fusion:
            if self.mm_fusion == "concat" and self.mm_fusion_type == "late":
                target_features *= 2
            elif self.mm_fusion == "adaptive":
                # adaptive_dim = target_tab_features * 2 if nb_tabular_data > 0 and nb_genetic_data > 0 else target_tab_features
                self.tab_weights = nn.Parameter(torch.ones(target_tab_features))
                self.img_weights = nn.Parameter(torch.ones(target_tab_features))
            elif self.mm_fusion == "bilinear":
                # bilinear_dim = target_tab_features * 2 if nb_tabular_data > 0 and nb_genetic_data > 0 else target_tab_features
                self.fuser = BilinearFusion(skip=1,use_bilinear=1, dim1=target_tab_features, dim2=target_tab_features, scale_dim1=4, scale_dim2=4, gate1=1, gate2=1, mmhid=target_tab_features*2)
                if self.mm_fusion_type == "late":
                    target_features *= 2

        self.mlp_head = nn.Linear(target_features, n_classes)
        initialize_weights(self)

    def forward(self, return_weights=False, **kwargs):
        x = kwargs['x_path']
        if self.fc:
            x = self.fc(x)
        x = torch.unsqueeze(x, 0)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if kwargs["x_tabular"] is not None:
            tab_feats = self.tabular_emb(kwargs["x_tabular"])
        
        # Early fusion
        if self.mm_fusion_type == "early":
            tab_feats = torch.unsqueeze(tab_feats, 1)
            if self.mm_fusion == "concat":
                x = torch.cat((x, tab_feats), dim=1)
            elif self.mm_fusion == "adaptive":
                x = tab_feats * self.tab_weights + self.img_weights * x
            elif self.mm_fusion == "multiply":
                x = torch.mul(tab_feats, x)
            else:
                raise NotImplementedError("Unknown fusion method")

        x = self.dropout(x)
        
        # Mid fusion
        if self.mm_fusion_type == "mid":
            tab_feats = torch.unsqueeze(tab_feats, 1)
            x, attn_weights = self.transformer(x, tab_feats)
        else:
            x, attn_weights = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        feats = self.to_feats(x)

        # Late fusion
        if self.mm_fusion_type == "late":
            if self.mm_fusion == "concat":
                feats = torch.cat((feats, tab_feats), dim=-1)
            elif self.mm_fusion == "adaptive":
                feats = tab_feats * self.tab_weights + self.img_weights * feats
            elif self.mm_fusion == "multiply":
                feats = torch.mul(tab_feats, feats)
            elif self.mm_fusion == "bilinear":
                feats = self.fuser(feats, tab_feats)
            else:
                raise NotImplementedError("Unknown fusion method")
        
        output = self.mlp_head(feats)
        if kwargs["return_feats"]:
            return output, feats, attn_weights
        
        return output

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# T = 1229
# tab = torch.randn(1, T).to("cuda")
# img = torch.randn(200, 768).to("cuda")
# model_dict = {
#     "target_features": 50,
#     "n_classes": 4,
#     "drop_out": .25,
#     "nb_tabular_data": T,
#     "path_input_dim": 768,
#     "mlp_depth": 5,
#     "mlp_type": "small",
#     "activation": "relu",
#     "skip": True,
#     "batch_norm": False,
#     "depth": 5, 
#     "mha_heads": 4, 
#     "model_dim": None, 
#     "mlp_dim": 64,
#     "dim_head": 16,
#     "pool": "cls"
# }
# model = ViT(mm_fusion="crossatt", mm_fusion_type="mid", **model_dict).to("cuda")
# out, att_weights = model(x_path=img, x_tabular=tab, return_feats=False, return_weights=True)
# print(model)
# print(out, att_weights.shape)

# for mm_fusion_type in ["early", "mid", "late"]:
#     for mm_fusion in ["concat", "adaptive", "multiply"]:
#         model = ViT(mm_fusion=mm_fusion, mm_fusion_type=mm_fusion_type, **model_dict).to("cuda")
#         out = model(x_path=img, x_tabular=tab, return_feats=False)
#         print(mm_fusion_type, mm_fusion, out)

# model = ViT(mm_fusion="bilinear", mm_fusion_type="late", **model_dict).to("cuda")
# out = model(x_path=img, x_tabular=tab, return_feats=False)
# print(out)
