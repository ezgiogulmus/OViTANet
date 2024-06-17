import torch
import torch.nn as nn
import math

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(SimpleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)


class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, img_vector, tab_vector):
        device = img_vector.device
        patches = None
        if len(img_vector.shape) > 2:
            patches = img_vector[:, 1:, :]
            img_vector = img_vector[:, 0, :]
            tab_vector = tab_vector[:, 0, :]
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(img_vector)
            z1 = self.linear_z1(img_vector, tab_vector) if self.use_bilinear else self.linear_z1(torch.cat((img_vector, tab_vector), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(img_vector)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(tab_vector)
            z2 = self.linear_z2(img_vector, tab_vector) if self.use_bilinear else self.linear_z2(torch.cat((img_vector, tab_vector), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(tab_vector)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)

        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, img_vector, tab_vector), 1)
        out = self.encoder2(out)
        if patches is not None:
            out = torch.cat([out.unsqueeze(1), patches], dim=1)
        return out
    
class SimpleFusion(nn.Module):
    def __init__(self, mm_fusion, dim, nb_of_vectors=2):
        super().__init__()
        self.mm_fusion = mm_fusion
        if self.mm_fusion == "adaptive":
            self.weights = nn.ParameterList([nn.Parameter(torch.ones(dim)) for _ in range(nb_of_vectors)])

        if self.mm_fusion == "bilinear":
            self.fuser = BilinearFusion(skip=1,use_bilinear=1, dim1=dim, dim2=dim, scale_dim1=4, scale_dim2=4, gate1=1, gate2=1, mmhid=dim)
            
    def forward(self, v):
        if self.mm_fusion == "concat":
            v = torch.cat(v, dim=1)
        elif self.mm_fusion == "adaptive":
            out = torch.zeros_like(v[0])
            for i in range(len(v)):
                out = out + self.weights[i] * v[i]
            v = out
        elif self.mm_fusion == "multiply":
            out = torch.ones_like(v[0])
            for i in range(len(v)):
                out = torch.mul(out, v[i])
            v = out
        elif self.mm_fusion == "bilinear":
            out = self.fuser(v[0], v[1])
            if len(v) > 2:
                for i in range(2, len(v)):
                    out = self.fuser(out, v[i])
            v = out
        return v
        

class Gated_Attention(nn.Module):
    def __init__(self, dim=1024, out_dim=1024, dropout=0.25):
        super(Gated_Attention, self).__init__()
        self.attention_a = nn.ModuleList([
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout)
        ])
        self.attention_b = nn.ModuleList([
            nn.Linear(dim, 256),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        ])
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.to_out = nn.Linear(256, out_dim, bias=False)
        
    def forward(self, img, tab):
        a = self.attention_a(img)
        b = self.attention_b(tab)
        A = torch.matmul(a, b.transpose(-1, -2))
        out = torch.matmul(A, b)
        out = self.to_out(out)
        return out, A
    

        
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., cross_attention=True):
        super().__init__()
        self.cross_attention = cross_attention
        self.heads = heads
        self.dim_head = dim_head
        self.scale = 1 / math.sqrt(self.dim_head)
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        if cross_attention:
            self.to_q = Gated_Attention(dim, inner_dim, dropout=dropout)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img, tab, return_weights=False):
        attn_dict = {}
        # Normalize both vectors
        img = self.norm(img)
        if self.cross_attention:
            tab = self.norm(tab)

        # Gated attention for query
        if self.cross_attention:
            q, gate_weights = self.to_q(img, tab)
            attn_dict["gate"] = gate_weights
        else:
            q = self.to_q(img)
        k, v = self.to_kv(img).chunk(2, dim=-1)
        b, n = q.shape[:2]
        q, k, v = map(lambda t: t.view(b, -1, self.heads, self.dim_head).transpose(1, 2), [q, k, v]) # B, H, N, D
        
        # Scaled dot product
        attn_output, attn_weight = self._scaled_dot_product(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        attn_dict["mha"] = attn_weight
        if return_weights:
            return self.to_out(attn_output), attn_dict
        return self.to_out(attn_output)

    def _scaled_dot_product(self, query, key, value):
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weight = self.attend(attn_weight)
        attn_weight = self.dropout(attn_weight)
        out = torch.matmul(attn_weight, value)
        return out, attn_weight
    
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


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., mm_fusion=None, hierarchical=False, fuser=None):
        super().__init__()
        self.mm_fusion = mm_fusion
        self.hierarchical = hierarchical and mm_fusion

        self.fuser = fuser
        
        self.norm = nn.LayerNorm(dim)
        layers = []
        for i in range(depth):
            layers.append(nn.ModuleList([
                MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, cross_attention=True if self.mm_fusion == "crossatt" and (self.hierarchical or i == depth-1) else False),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, tab):
        attn_weights = None 
        return_weights = False 
        fusion = True if self.hierarchical else False
        for i, (attn, ff) in enumerate(self.layers):            
            if i == len(self.layers) - 1:
                return_weights = True
                fusion = True
            if fusion:
                if self.fuser is not None:
                    x = self.fuser([x, tab])
                att_out = attn(x, tab=tab if self.mm_fusion =="crossatt" else None, return_weights=return_weights)
            else:
                att_out = attn(x, tab=None, return_weights=return_weights)
            if isinstance(att_out, tuple):
                att_out, attn_weights = att_out
                
            x = att_out + x
            x = ff(x) + x
        return self.norm(x), attn_weights
    

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
