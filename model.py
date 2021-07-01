import torch
from torch import nn, einsum
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
import math
from functools import partial

def full_attention(query, key, value, causal=False, dropout=0.0):
    """
    Scale Dot-Product Attention (論文Fig.2)

    inputs:
      - query (torch.tensor) (B, h, n, d)
      - key (torch.tensor) (B, h, n, d)
      - value (torch.tensor) (B, h, n, d)
      - causal (bool) : Trueの時、時間マスク(三角行列)を使用
      - dropout (float) : ドロップアウトの割合(使用するなら)
    
    return:
      - out (torch.tensor) (B, h, n, d)
    """
    device = key.device
    B_k, h_k, n_k, d_k = key.shape
    B_q, h_q, n_q, d_q = query.shape
    # import pdb; pdb.set_trace()

    scale = einsum("bhqd,bhkd->bhqk", query, key)/math.sqrt(d_k)

    if causal:
        # マスクを作る(下三角行列)
        ones = torch.ones(B_k, h_k, n_q, n_k).to(device)
        mask = torch.tril(ones)
        scale = scale.masked_fill(mask == 0, -1e9)# -infで埋めるイメージ。めちゃめちゃ確率小さくなる
    atn = F.softmax(scale, dim=-1)
    if dropout is not None:# ここにはさむべき？？
        atn = F.dropout(atn, p=dropout)   
    # out = torch.matmul(atn, value)
    out = einsum("bhqk,bhkd->bhqd", atn, value)
    return out

def to_eachhead(x, head_num, split_num=3):
    """
    入力テンソルをsplit_num個に分割(3の時qvk)して、ヘッドに分割

    (B, n, D) -> (B, n, d) x split_num -> (B, h, n, d')

    ただし、D = d x split_num

    - inputs
        - x (torch.tesor) : (B, n, 3d) output of self.qvk
        - head_num : head数
        - split_num : 分割数、qvkに分割する場合は、split_num=3
    - outpus
        - out (list)
            - out = [q, v, ...(split num)]
                - q (torch.tensor) : (B, h, n, d')
                - v (torch.tensor) : (B, h, n, d')
                - k (torch.tensor) : (B, h, n, d')
                    - ただしd'はマルチヘッドアテンションを行う時の次元数
    """
    B, n, pre_d = x.shape
    new_d = pre_d//split_num
    assert pre_d%split_num == 0, f"have to be multiple of {split_num}"
    assert new_d%head_num == 0, "dim must be divided by head_num"

    tpl = torch.chunk(x, split_num, dim=2)
    out = []
    for t in tpl:
        out.append(t.reshape(B, n, head_num, new_d//head_num).transpose(1,2))
    return out

def concat_head(x):
    """
    ヘッドをもとに戻す

    - inputs
        - x (torch.tensor) : (B, h, n, d')
    - outputs
        - out (torch.tensor) : (B, n, d) (d = d' x h)
    """
    B, h, n, _d = x.shape
    out = x.transpose(1,2).reshape(B, n, _d*h)
    return out

class PositionalEncoding(nn.Module):
    """
    位置エンコーディング

    args:
      - d_model (int) : ベクトルの次元数
      - device
      - max_len (int) : 許容しうる最大の長さの文章
    """
    def __init__(self, d_model, device="cpu", max_len = 100):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0,d_model,2)
        den = 1/torch.pow(torch.ones(int(d_model/2))*max_len,2*tmp/d_model)
        den = den.unsqueeze(0)
        self.pe[:,0::2] = torch.sin(torch.matmul(pos,den))
        self.pe[:,1::2] = torch.cos(torch.matmul(pos,den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        # scale = self.pe.shape[1]**0.5
        return x + self.pe[:x.shape[1],:]

class PreLayer(nn.Module):
    def __init__(self, hid, d_model, drop_out=0.0, in_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)

    def forward(self, x):
        out = self.linear(x)
        return out

class PostLayer(nn.Module):
    def __init__(self, dim, vocab_num, hid_dim, dropout_ratio):
        super().__init__()
        self.linear = nn.Linear(dim, vocab_num)
    def forward(self,x):
        out = self.linear(x)
        return out

class Encoder(nn.Module):
    """
    コピータスクのエンコーダ
    EncoderLayerを所望の数積み重ねる

    - args:
        - depth : 層の数
        - dim : 潜在次元数
        - head_num : ヘッド数
        - attn_type : linear -> LinearAttention / full -> Vannila
        - ff_hidnum : feedforwardにおける潜在次元数
        - dropout_ratio (float) : 

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - x : (torch.tensor) : (B, N, D)
    """
    def __init__(self, depth, dim, head_num, attn_type="linear", ff_hidnum=2048, dropout_ratio=0.2):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, attn_type, head_num, ff_hidnum, dropout_ratio) for i in range(depth)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer(nn.Module):
    """
    コピータスクのエンコーダレイヤー
    selfattention -> feedforward
    residual passとそれに伴ったLayerNormを実装

    - args:
        - dim : 潜在次元数
        - attn_type : attentionのタイプ
        - head_num : ヘッド数
        - ff_hidnum (int) : feedforwardでの隠れ層の次元
        - dropout_ratio (float) : 

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num, ff_hidnum, dropout_ratio):
        super().__init__()
        self.dor = dropout_ratio
        self.mhsa = MultiHeadSelfAttention(dim, attn_type, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = torch.clone(x)
        out = self.mhsa(x)
        out = F.dropout(out, p=self.dor) + res
        out = self.ln1(out)
        res = torch.clone(out)
        out = self.ff(out)
        out = F.dropout(out, p=self.dor) + res
        out = self.ln2(out)
        return out

class FeedForward(nn.Module):
    """
    feedforwad module. 2層のaffine層

    - args:
        - dim (int)
        - hid_dim (int)

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, hid_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hid_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, dim, bias=True)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """
    multiheadselfattention
    head増やす(B, H, N, D) -> selfattention function -> output

    - args:
        - dim (int) : 
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : 
    
    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        if attn_type == "full":
            self.mhsa = full_attention
        else:
            raise NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))
    
    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out

class MultiHeadCausalAttention(nn.Module):
    """
    Causal attentionをやります。
    head増やす(B, H, N, D) -> causalattention function -> output

    - args:
        - dim (int) : 特徴次元数
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : ヘッド数
    
    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        if attn_type == "full":
            self.mhca = partial(full_attention, causal=True)
        else:
            raise NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))

    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhca(q, k, v)
        out = concat_head(out)
        return out

class MultiHeadSourceAttention(nn.Module):
    """
    source attention. this is for attention using output of encoder(memory). 

    - args:
        - dim (int) : 特徴次元数
        - attn_type (str) : linear -> LinearAttention / full -> Vannila
        - head_num (int) : ヘッド数

    - inputs:
        - x (torch.tensor) : (B, N, D) input tensor
        - memory (torch.tensor) : (B, N, D) output of encoder

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num):
        super().__init__()
        self.to_kv = nn.Linear(dim, dim*2)
        self.to_q = nn.Linear(dim, dim)
        self.make_head_kv = partial(to_eachhead, head_num=head_num, split_num=2)
        self.make_head_q = partial(to_eachhead, head_num=head_num, split_num=1)
        if attn_type == "full":
            self.mhsa = full_attention
        else:
            raise NotImplementedError("attention type of {} is not defined. Please set linear or full".format(attn_type))

    def forward(self, x, memory):
        mem = self.to_kv(memory)
        x = self.to_q(x)
        k, v = self.make_head_kv(mem)
        q = self.make_head_q(x)[0]
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out

class Decoder(nn.Module):
    """
    コピータスクのデコーダ
    DecoderLayerを所望の数積み重ねる

    - args:
        - depth : 層の数
        - dim : 潜在次元数
        - head_num : ヘッド数
        - attn_type : linear -> LinearAttention / full -> Vannila
        - ff_hidnum : feedforwardにおける潜在次元数
        - dropout_ratio (float) : 

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - x : (torch.tensor) : (B, N, D)
    """
    def __init__(self, depth, dim, head_num, attn_type, ff_hidnum, dropout_ratio=0.2):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, attn_type, head_num, ff_hidnum, dropout_ratio) for i in range(depth)])
    
    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x

class DecoderLayer(nn.Module):
    """
    (self)causalattention -> sourceattention -> feedforward
    residual passとそれに伴ったLayerNormを実装

    - args:
        - dim (int) : 潜在次元数
        - attn_type (str) : attentionのタイプ
        - head_num (int) : ヘッド数
        - ff_hidnum (int) : feedforwardでの隠れ層の次元
        - dropout_ratio (float) : 

    - inputs:
        - x (torch.tensor) : (B, N, D)

    - outputs:
        - out (torch.tensor) : (B, N, D)
    """
    def __init__(self, dim, attn_type, head_num, ff_hidnum, dropout_ratio):
        super().__init__()
        self.dor = dropout_ratio
        self.mhca = MultiHeadCausalAttention(dim, attn_type, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSourceAttention(dim, attn_type, head_num)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        res = torch.clone(x)
        out = self.mhca(x)
        out = F.dropout(out, p=self.dor) + res
        out = self.ln1(out)
        res = torch.clone(out)
        out = self.mhsa(out, memory)
        out = F.dropout(out, p=self.dor) + res
        out = self.ln2(out)
        res = torch.clone(out)
        out = self.ff(out)
        out = F.dropout(out, p=self.dor) + res
        out = self.ln3(out)
        return out

class Transformer(nn.Module):
    """
    Transformerモデル。(入力長固定の場合を考えており、paddingに対するマスクは使用していない。)
    position -> encoder -> decoder -> Postlayer

    - args:
        - device (str) : cpu or gpu name
        - d_model (int) : 潜在次元数
        - in_dim (int) : 入力次元数(月などの補足データがある時は1ではなくなる)
        - attn_type (str) : type of attention ("full")
        - N_enc (int) : number of encoderlayer
        - N_dec (int) : number of decoderlayer
        - h_enc (int) : number of multihead in encoder
        - h_dec (int) : number of multihead in decoder
        - ff_hidnum (int) : hiddne dim at FeedForwardNetwork
        - hid_pre (int) : hidden dim at PreLayer 
        - hid_post (int) : hidden dim at PostLayer 
        - dropout_pre (float) : dropout ratio at PreLayer
        - dropout_post (float) : dropout ratio at PostLayer
        - dropout_model (float) : dropout ratio at encoder and decoder which is adopted to the output of sublayer(ex. SelfAttention or FeedForward)
        - dropout_post (float) : dropout ratio at post layer

    - inputs:
        - x (torch.tensor) : (B, len_x) or (B, len_x, D)
        - y (torch.tensor) : (B, len_y) or (B, len_y, D)

    - outputs:
        - out (torch.tensor) : (B, len_gen)
    """
    def __init__(self, device, d_model, in_dim, attn_type, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model):
        super().__init__()
        self.device = device
        self.x_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim)
        self.y_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim)
        self.pos = PositionalEncoding(d_model, device=device) # src, tgt共通
        self.enc = Encoder(N_enc,d_model, h_enc, attn_type, ff_hidnum, dropout_model)
        self.dec = Decoder(N_dec,d_model, h_dec, attn_type, ff_hidnum, dropout_model)
        self.post = PostLayer(d_model, 1, hid_post, dropout_post)

    def forward(self, x, y):
        # import pdb; pdb.set_trace()
        x_emb = self.x_pre(x)
        y_emb = self.y_pre(y)
        x_emb_pos = self.pos(x_emb)
        y_emb_pos = self.pos(y_emb)
        memory = self.enc(x_emb_pos)
        out = self.dec(y_emb_pos, memory)
        out = self.post(out)
        out = out.squeeze(-1)
        return out

    def generate(self, x, forcast_step, y_start):
        """
        自己回帰的に生成する.
        所望の長さになったら終了するように実装した。

        - args:
            - x (torch.tensor) (B, N) : 入力時系列
            - forcast_step (int) : 予測する個数。
            - y_start (troch.tensor) (B,) : x_tのデータ(x_2以降を予測する時、x_1のデータを初期値として利用)
        """
        device = x.device
        B, N, D = x.shape
        x = self.x_pre(x)
        x = self.pos(x)
        z = self.enc(x)
        # y_start = y_start.unsqueeze(-1)
        y = y_start
        # import pdb; pdb.set_trace()
        for i in range(forcast_step):
            # mask = make_mask(y.shape[1])
            y_pred = self.y_pre(y)
            y_pred = self.pos(y_pred)
            y_pred = self.dec(y_pred, z)
            # import pdb; pdb.set_trace()
            y_pred = self.post(y_pred)# (B, N, 1)
            y = torch.cat([y, y_pred[:,[-1],:]], dim=1)
        y_pred = y_pred.squeeze(-1)
        return y_pred

