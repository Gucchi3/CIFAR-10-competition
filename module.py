# @file module.py
# @brief Blocks and optimizers for main.py

from collections.abc import Callable
from torch           import nn
from typing          import overload

import torch



class DConv2d(nn.Module):
    '''
    #* チャネルごとの3x3畳み込みと1x1畳み込みを続けて行う。
    '''
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class RBlock(nn.Module):
    '''
    #* 3x3畳み込み2回の結果を入力に足す。
    '''
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)


class ConvNeXtBlock(nn.Module):
    '''
    #* 大きいdepthwise畳み込みと、広げて戻す1x1畳み込みを使う残差ブロック。
    '''
    
    def __init__(self, channels: int, expand: int = 4, k: int = 7) -> None:
        super().__init__()
        self.dwconv  = nn.Conv2d(channels, channels, kernel_size=k, padding=k // 2, groups=channels)
        self.norm    = nn.GroupNorm(1, channels)
        self.pwconv1 = nn.Conv2d(channels, channels * expand, kernel_size=1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * expand, channels, kernel_size=1)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


class SEBlock(nn.Module):
    '''
    #* チャネルごとの重みを計算して、入力にかける。
    '''
    
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class SAttention(nn.Module):
    '''
    #* 画像全体の位置どうしでattentionを計算し、入力に足す。
    '''
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key   = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q    = self.query(x).flatten(2).transpose(1, 2)
        k    = self.key(x).flatten(2)
        v    = self.value(x).flatten(2).transpose(1, 2)
        attn = torch.softmax(q @ k / c ** 0.5, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + out


class WAttention(nn.Module):
    '''
    #* 画像を window_size 四方の窓に分け、窓の中だけでattentionを計算する。
    #! 縦横の大きさは window_size で割り切れること。
    '''
    
    def __init__(self, channels: int, window_size: int = 4) -> None:
        super().__init__()
        self.ws  = window_size
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws         = self.ws
        
        # 窓ごとに並べ替える
        q, k, v = [t.reshape(b, c, h // ws, ws, w // ws, ws).permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c) for t in self.qkv(x).chunk(3, dim=1)]
        
        # 窓の中でattentionを計算して、元の形に戻す
        attn = torch.softmax(q @ k.transpose(1, 2) / c ** 0.5, dim=-1)
        out  = (attn @ v).reshape(b, h // ws, w // ws, ws, ws, c).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return x + out


class Muon(torch.optim.Optimizer):
    '''
    #* Conv重み(4次元)用のMuon。勾配を2次元に直してNewton-Schulzで直交化する。
    #? build_module_muon_optimizer() から使う。
    #! 4次元以外の重み(Linear、BN、bias)はAdamWに渡す。
    '''
    
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum))
        return None
    
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # closureがあれば損失を計算し直す
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 重みを1つずつ更新する
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Nesterovモーメンタム
                buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g   = p.grad.add(buf, alpha=group["momentum"]).reshape(p.size(0), -1)
                
                # Newton-Schulzで直交化
                x = g / (g.norm() + 1e-7)
                t = x.size(0) > x.size(1)
                x = x.T if t else x
                for _ in range(5):
                    a = x @ x.T
                    x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
                x = x.T if t else x
                
                # 形を戻して更新
                p.add_(x.view_as(p), alpha=-group["lr"] * max(1, g.size(0) / g.size(1)) ** 0.5)
        return loss


def build_module_adam_optimizer(model: nn.Module) -> list[torch.optim.Optimizer]:
    '''
    #* Adam(lr=1e-3)を作る。
    '''
    
    return [torch.optim.Adam(model.parameters(), lr=1e-3)]


def build_module_adamw_optimizer(model: nn.Module) -> list[torch.optim.Optimizer]:
    '''
    #* AdamW(lr=1e-3, weight_decay=0.05)を作る。
    '''
    
    return [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)]


def build_module_sgd_optimizer(model: nn.Module) -> list[torch.optim.Optimizer]:
    '''
    #* SGD(lr=0.05, momentum=0.9)を作る。
    '''
    
    return [torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)]


def build_module_muon_optimizer(model: nn.Module) -> list[torch.optim.Optimizer]:
    '''
    #* Conv重みはMuon(lr=0.02)、それ以外はAdamW(lr=1e-3)にする。
    #! Conv層が1つもないモデルでは使えない。
    '''
    
    muon_params  = [p for p in model.parameters() if p.ndim == 4]
    adamw_params = [p for p in model.parameters() if p.ndim != 4]
    return [Muon(muon_params, lr=0.02), torch.optim.AdamW(adamw_params, lr=1e-3)]


OPTIMIZER_REGISTRY = {
    "adam" : build_module_adam_optimizer,
    "adamw": build_module_adamw_optimizer,
    "sgd"  : build_module_sgd_optimizer,
    "muon" : build_module_muon_optimizer,
}


def build_module_optimizer(name: str, model: nn.Module) -> list[torch.optim.Optimizer]:
    '''
    #* 名前からoptimizerのリストを作る。
    #? main.py から呼ぶ。
    #! 学習ループでは、リストの全部に zero_grad() と step() を呼ぶ。
    '''
    
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"unknown optimizer: {name}. choose from {list(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](model)
