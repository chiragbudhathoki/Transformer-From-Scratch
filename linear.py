import torch
import torch.nn as nn
import einops
import math
import numpy as np

import os 
import typing
from collections.abc import Callable, Iterable
from typing import Optional
import pickle
from types import SimpleNamespace

import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    # Base vocab: 256 bytes, then special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # --- REPLACED PART: Stream in 10 MB chunks instead of f.read() ---
    if special_tokens:
        split_pat = "|".join(
            re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
        )
        max_spec_len = max(len(t) for t in special_tokens)
    else:
        split_pat = None
        max_spec_len = 0

    word_freq = Counter()
    buffer = ""
    bytes_read = 0
    next_log = 200 * 1024 * 1024

    with open(input_path, encoding="utf-8") as f:
        while True:
            chunk = f.read(10 * 1024 * 1024)
            if not chunk:
                if buffer:
                    segs = re.split(split_pat, buffer) if split_pat else [buffer]
                    for seg in segs:
                        for m in re.finditer(PAT, seg):
                            word_freq[tuple(m.group().encode("utf-8"))] += 1
                break

            bytes_read += len(chunk)
            buffer += chunk

            if split_pat:
                segs = re.split(split_pat, buffer)
                tail = segs.pop()
                if len(tail) > max_spec_len + 64:
                    cut = len(tail) - (max_spec_len + 64)
                    segs.append(tail[:cut])
                    buffer = tail[cut:]
                else:
                    buffer = tail
            else:
                segs = [buffer[:-64]]
                buffer = buffer[-64:]

            for seg in segs:
                for m in re.finditer(PAT, seg):
                    word_freq[tuple(m.group().encode("utf-8"))] += 1

            if bytes_read >= next_log:
                print(f"      Pre-tokenized {bytes_read / 1e6:.0f} MB...")
                next_log += 200 * 1024 * 1024

    print(f"      Found {len(word_freq):,} unique words. Running BPE merges...")
    # --- END OF REPLACED PART ---

    # --- KEEP YOUR EXISTING MERGE CODE BELOW ---
    words = [list(w) for w in word_freq]
    freqs = list(word_freq.values())

    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for idx, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_counts[p] += freqs[idx]
            pair_to_words[p].add(idx)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(
            pair_counts,
            key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])),
        )
        a, b = best
        merges.append((vocab[a], vocab[b]))
        if len(merges) % 1000 == 0:
            print(f"      Completed {len(merges)} / {vocab_size - len(special_tokens) - 256} merges...")

        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]

        for idx in list(pair_to_words[best]):
            w, f = words[idx], freqs[idx]

            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    del pair_counts[p]

            new_w, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new_w.append(new_id)
                    i += 2
                else:
                    new_w.append(w[i])
                    i += 1
            words[idx] = new_w

            for p in zip(new_w, new_w[1:]):
                pair_counts[p] += f
                pair_to_words[p].add(idx)

        pair_to_words.pop(best, None)

    return vocab, merges


def build_tokenizer(vocab, merges, special_tokens=None):
    """Return an object with .encode(text), .encode_iterable(iterable), .decode(ids)."""
    vocab = dict(vocab)
    special_tokens = list(special_tokens) if special_tokens else []

    # Make sure every special token is in the vocab
    existing = set(vocab.values())
    for tok in special_tokens:
        tb = tok.encode("utf-8")
        if tb not in existing:
            vocab[len(vocab)] = tb
            existing.add(tb)

    bytes_to_id = {b: i for i, b in vocab.items()}
    merge_ranks = {pair: r for r, pair in enumerate(merges)}
    cache = {}
    specials = set(special_tokens)

    if special_tokens:
        pattern = "(" + "|".join(
            re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)
        ) + ")"
        special_re = re.compile(pattern)
    else:
        special_re = None

    def bpe(word):
        if word in cache:
            return cache[word]
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = merge_ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            parts[best_i:best_i + 2] = [parts[best_i] + parts[best_i + 1]]
        ids = [bytes_to_id[p] for p in parts]
        cache[word] = ids
        return ids

    def encode(text):
        ids = []
        chunks = special_re.split(text) if special_re else [text]
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in specials:
                ids.append(bytes_to_id[chunk.encode("utf-8")])
                continue
            for m in re.finditer(PAT, chunk):
                ids.extend(bpe(m.group().encode("utf-8")))
        return ids

    def encode_iterable(iterable):
        for piece in iterable:
            yield from encode(piece)

    def decode(ids):
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    return SimpleNamespace(encode=encode, encode_iterable=encode_iterable, decode=decode)


def load_tokenizer(vocab_path, merges_path, special_tokens=None):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    return build_tokenizer(vocab, merges, special_tokens)


class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device = None,dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        weight = torch.empty((out_features,in_features),device=device,dtype=dtype)
        sd = math.sqrt(2/(in_features + out_features))
        a = -3 * sd
        b = 3 * sd
        weight = nn.Parameter(torch.nn.init.trunc_normal_ (weight,mean = 0,std = sd,a = a,b = b))
        self.weight = weight

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = einops.einsum(x,self.weight,"... in_features,out_features in_features -> ... out_features")
        return x




class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device = None,dtype = None): #num_embeddings = vocab_size, embedding_dim = d_model
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
       
        weight = torch.empty((num_embeddings,embedding_dim),device = device,dtype = dtype)
        weight = nn.Parameter(nn.init.trunc_normal_(weight,mean = 0,std = 1,a =-3, b = 3))
        
        self.weight = weight

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        token_ids = self.weight[token_ids]
        return token_ids



class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float = 1e-5,device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        weight = torch.ones((d_model,),device = device,dtype = dtype)
        weight = nn.Parameter(weight)
        self.weight = weight

    def forward(self,x : torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = torch.sqrt(torch.mean(x**2,dim = -1,keepdim = True)+self.eps)
        result = (x/RMS) *self.weight
        result = result.to(in_dtype)
        return result



class positionwise_feedforward(nn.Module):
    def __init__(self,d_model:int,d_ff:int|None = None,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        
        if d_ff is None:
            raw_dff = 8/3 * d_model
            d_ff = round(raw_dff/64) * 64
        else:
            d_ff = d_ff
        
        self.d_ff = d_ff

        w1_weight = Linear(d_model,d_ff,device = device,dtype=dtype)
        w2_weight = Linear(d_ff,d_model,device = device,dtype=dtype)
        w3_weight = Linear(d_model,d_ff,device = device,dtype=dtype)

        self.w1_weight = w1_weight
        self.w2_weight = w2_weight
        self.w3_weight = w3_weight

    def forward(self,x:torch.Tensor)->torch.Tensor:
        w1x = self.w1_weight(x)
        SiLU = w1x * torch.sigmoid(w1x)
        w3x = self.w3_weight(x)
        GLU = SiLU * w3x
        SwiGLU = self.w2_weight(GLU)
        return SwiGLU
    

class RoPE(nn.Module):
    def __init__(self, theta:float,d_k:int,max_seq_len : int,device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        k = torch.arange(self.d_k //2,device = device)
        self.k = k
        wk = theta**(-2*k/d_k)
        self.wk = wk
        i = torch.arange(max_seq_len,device = device,dtype = torch.float32)
        angles = torch.outer(i,wk)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)

    def forward(self,x : torch.Tensor,token_position : torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_position]
        sin = self.sin[token_position]
        x_odd = x[...,1::2]
        x_even = x[...,::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        stacked = torch.stack((rot_even,rot_odd),dim = -1)
        result = torch.flatten(stacked,start_dim = -2,end_dim = -1)
        return result

def softmax(in_features,dim):
    m = torch.max(in_features,dim,keepdim=True).values
    vi = in_features - m
    exp = torch.exp(vi)
    summed_exp = torch.sum(exp,dim,keepdim=True)
    probs = exp/summed_exp
    return probs

def scaled_dot_product_attention(Q,K,V,mask =None):
    d_k = Q.shape[-1]
    QK = einops.einsum(Q,K,"... queries d_k , ... keys d_k -> ... queries keys")
    masked_QK = QK.masked_fill(mask == 0, float('-inf')) if mask is not None else QK
    div = softmax(masked_QK/math.sqrt(d_k),dim = -1)
    result = einops.einsum(div,V,"... queries keys , ... keys d_v -> ... queries d_v")
    return result

 
class multihead_self_attention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len = None,theta = None,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        dk = d_model//num_heads
        q_proj_weight = Linear(d_model,d_model,device= device,dtype = dtype)
        k_proj_weight = Linear(d_model,d_model,device= device,dtype = dtype)
        v_proj_weight = Linear(d_model,d_model,device= device,dtype = dtype)
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight

        o_proj_weight = Linear(d_model,d_model,device= device,dtype = dtype)
        self.o_proj_weight = o_proj_weight

        self.dk = dk
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        if max_seq_len is not None and theta is not None:
            rope_qk = RoPE(theta = self.theta,d_k = self.dk,max_seq_len = self.max_seq_len,device = self.device)
            self.rope_qk = rope_qk
        else:
            self.rope_qk = None
        
            
    def forward(self,x:torch.Tensor,mask:torch.Tensor = None,token_position:torch.Tensor = None)->torch.Tensor:
        q,k,v = self.q_proj_weight(x),self.k_proj_weight(x),self.v_proj_weight(x)
        
        q = einops.rearrange(q,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads = self.num_heads,d_k = self.dk)
        k = einops.rearrange(k,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads = self.num_heads,d_k = self.dk)
        v = einops.rearrange(v,"... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",num_heads = self.num_heads,d_k = self.dk)
        if self.rope_qk is not None :
            if token_position is None:
                token_position = torch.arange(x.shape[-2], device=x.device)
            pos = token_position.unsqueeze(-2)
            q = self.rope_qk(q,pos)
            k = self.rope_qk(k,pos)
    
        if mask is None:
            mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], device=x.device, dtype=torch.bool))

        out = scaled_dot_product_attention(q,k,v,mask)
        out= einops.rearrange(out,"... num_heads seq_len d_k -> ... seq_len (num_heads d_k)",num_heads = self.num_heads,d_k = self.dk)
        return self.o_proj_weight(out)


class transformer_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,max_seq_len,theta,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        

        ln1 = RMSNorm(d_model,eps=1e-5,device=device,dtype=dtype)
        attn = multihead_self_attention(d_model,num_heads,max_seq_len,theta,device=device,dtype=dtype)
        ln2= RMSNorm(d_model,eps=1e-5,device = device,dtype= dtype)
        ffw = positionwise_feedforward(d_model,d_ff,device,dtype)

        self.ln1 = ln1
        self.attn = attn
        self.ln2 = ln2
        self.ffw = ffw
    def forward(self,x:torch.Tensor,mask:torch.Tensor|None = None,token_position:torch.Tensor|None = None)->torch.Tensor :
        z = x + self.attn(self.ln1(x),mask,token_position)
        y = z + self.ffw(self.ln2(z))
        return y


class transformer_lm(nn.Module):
    def __init__(self,vocab_size,d_model,num_layers,num_heads,d_ff,max_seq_len,theta,device = None,dtype = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        embed = Embedding(vocab_size,d_model,device,dtype)
        self.embed = embed
        block = nn.ModuleList([transformer_block(d_model,num_heads,d_ff,max_seq_len,theta,device,dtype) for _ in range(num_layers)])
        self.block = block
        ln = RMSNorm(d_model,eps=1e-5,device=device,dtype=dtype)
        self.ln = ln    
        lm_head = Linear(d_model,vocab_size,device,dtype)
        self.lm_head = lm_head

    def forward(self,x:torch.Tensor,mask:torch.Tensor| None = None,token_position:torch.Tensor| None = None)-> torch.Tensor:
        embedding = self.embed(x)
        for layer in range(self.num_layers):
            embedding = self.block[layer](embedding,mask,token_position)
        out = self.ln(embedding)
        logits = self.lm_head(out)
        return logits

def cross_entropy(input:torch.Tensor,target:torch.Tensor):
    m = torch.max(input,-1,keepdim = True).values
    vi = input - m
    exp = torch.exp(vi)
    target = torch.flatten(target)
    vi = torch.flatten(vi,start_dim = 0,end_dim = -2)
    vi = vi[torch.arange(vi.shape[0]),target]
    summed_exp = torch.sum(exp,-1,keepdim = True)
    summed_exp = torch.flatten(summed_exp,start_dim = 0,end_dim = -1)
    logits = -vi + torch.log(summed_exp)
    return torch.mean(logits)

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr = 1e-3,betas= ((0.9,0.999)),eps = 1e-8,weight_decay=0.01):
        if lr<0 or eps<0 or weight_decay<0 or betas[0]<0.0 or betas[0]>=1.0 or betas[1]<0.0 or betas[1]>=1.0:
            raise ValueError("Invalid Hyperparameters")
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)

    def step(self,closure:Optional[Callable] = None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                betas = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    t = state.get("t",0)
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p)

                    grad = p.grad
                    b1,b2 = betas
                    state["m"] = b1 * state["m"] + (1 - b1) * grad
                    state["v"] = b2 * state["v"] + (1 - b2) * grad**2
                   
                    m = state["m"]/(1-b1**(t+1))
                    v = state["v"]/(1-b2**(t+1))
                    p = p.add_(p,alpha = -lr * weight_decay)
                    p = p.addcdiv_(m,v.sqrt() + eps, value = -lr)
                    state["t"] = t + 1
                    
            return loss



def lr_cosine_scheduler(t,alpha_max,alpha_min,Tw,Tc):
    if t < Tw:
        alpha_t = (t/Tw)*alpha_max
    elif Tw <= t and t <= Tc:
        alpha_t = alpha_min + 1/2*(1 + math.cos((t-Tw)/(Tc-Tw)*math.pi))*(alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t

def gradient_clipping(parameters,max_l2_norm):
    total_norm = 0.0
    params = list(parameters)
    for p in params:
        
        if p.grad is not None:
            g = p.grad
            norm = g.pow(2).sum()
            total_norm += norm
    total_norm = total_norm ** 0.5
    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    return None

def get_batch(x:np.ndarray,batch_size:int,context_length:int,device:str):
    s_idx = np.random.randint(0,len(x) - context_length,size = batch_size)
    input = np.stack([x[s:s + context_length]for s in s_idx])
    target = np.stack([x[s+1:s + context_length + 1]for s in s_idx])
    input = torch.tensor(input,device=device,dtype = torch.long)
    target = torch.tensor(target,device = device,dtype = torch.long)
    return (input,target)


def save_checkpoint(model:nn.Module,optimizer:torch.optim.Optimizer,iteration:int,out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    obj = {
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "iteration":iteration}
    torch.save(obj,out)
    return out


def load_checkpoint(src:str | os.PathLike | typing.BinaryIO | typing.IO[bytes],model:nn.Module,optimizer:torch.optim.Optimizer)->int:
    src = torch.load(src, weights_only=False)
    model.load_state_dict(src["model"])
    optimizer.load_state_dict(src["optimizer"])
    return int(src["iteration"])
