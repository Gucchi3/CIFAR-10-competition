# モデル

## 畳み込み系

### 1. Conv2d
- 長所: 局所的な特徴を少ないパラメータで捉えられる
- 短所: 1層だけでは広い範囲を見られない

$$
y_{c,i,j} = b_c + \sum_{k} \sum_{m,n} W_{c,k,m,n} \, x_{k,\,i+m,\,j+n}
$$

```python
nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
```

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
```

### 2. Residual Block
- 長所: 層を深くしても学習が崩れにくい
- 短所: 入出力のチャネル数をそろえる必要がある

$$
y = \mathrm{ReLU}\left(x + \mathcal{F}(x)\right)
$$

$$
\mathcal{F}(x) = \mathrm{BN}\left(W_2 * \mathrm{ReLU}\left(\mathrm{BN}(W_1 * x)\right)\right)
$$

```python
RBlock(channels=24)
```

```python
class RBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(x + out)
```

### 3. ConvNeXt Block
- 長所: パラメータの割に精度が高い
- 短所: 7×7とチャネル4倍で計算が重め

$$
y = x + W_2 \, \mathrm{GELU}\left(W_1 \, \mathrm{Norm}\left(\mathrm{DWConv}_{7 \times 7}(x)\right)\right)
$$

```python
ConvNeXtBlock(channels=24, expand=4, k=7)
```

```python
class ConvNeXtBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.dwconv  = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm    = nn.GroupNorm(1, channels)
        self.pwconv1 = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * 4, channels, kernel_size=1)

    def forward(self, x):
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return x + out
```

### 4. Depthwise Separable Conv
- 長所: パラメータと計算量が数分の1になる
- 短所: 通常のConvより表現力がやや落ちる

$$
y = W_{\mathrm{pw}} * \mathrm{DWConv}_{3 \times 3}(x)
$$

```python
DConv2d(in_channels=12, out_channels=24)
```

```python
class DConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

### 5. SE Block
- 長所: ほぼタダで重要なチャネルを強調できる
- 短所: 位置 (どこが大事か) は見ない

$$
s = \sigma\left(W_2 \, \mathrm{ReLU}\left(W_1 \, \mathrm{GAP}(x)\right)\right), \quad y = s \odot x
$$

```python
SEBlock(channels=24, reduction=4)
```

```python
class SEBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w
```

## Attention系

### 6. Self Attention
- 長所: 画像全体の関係を一度に見られる
- 短所: 画素数の2乗で重くなる

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

```python
SAttention(channels=24)
```

```python
class SAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, kernel_size=1)
        self.key   = nn.Conv2d(channels, channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2)
        v = self.value(x).flatten(2).transpose(1, 2)

        attn = torch.softmax(q @ k / c ** 0.5, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(b, c, h, w)

        return x + out
```

### 7. Window Attention
- 長所: 大きい特徴マップにも使える
- 短所: 窓をまたいだ情報が混ざらない

$$
y = x + \mathrm{merge}\left(\left\lbrace \mathrm{softmax}\left(\frac{Q_w K_w^\top}{\sqrt{d}}\right)V_w\right\rbrace_{w}\right)
$$

```python
WAttention(channels=24, window_size=4)
```

```python
class WAttention(nn.Module):

    def __init__(self, channels, window_size=4):
        super().__init__()
        self.ws  = window_size
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.ws

        q, k, v = self.qkv(x).chunk(3, dim=1)

        def to_windows(t):
            t = t.reshape(b, c, h // ws, ws, w // ws, ws)
            return t.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)

        q, k, v = to_windows(q), to_windows(k), to_windows(v)

        attn = torch.softmax(q @ k.transpose(1, 2) / c ** 0.5, dim=-1)
        out  = attn @ v

        out = out.reshape(b, h // ws, w // ws, ws, ws, c)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return x + out
```

## 活性化関数

### 8. ReLU
- 長所: 単純で速い
- 短所: 負の側で勾配が0になり、出力が止まるニューロンが出る

$$
\mathrm{ReLU}(x) = \max(0, x)
$$

```python
nn.ReLU()
```

### 9. SiLU
- 長所: なめらかで、ReLUより精度が上がることがある
- 短所: ReLUより少し計算が重い

$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

```python
nn.SiLU()
```

### 10. GELU
- 長所: Transformer系と相性がいい
- 短所: ReLUより計算が重く、小さいCNNでは差が出にくい

$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

```python
nn.GELU()
```

## プーリング

### 11. Max Pooling
- 長所: 縦横を半分にして計算を減らせる
- 短所: 情報を捨てる

$$
y_{c,i,j} = \max_{0 \le m,n < 2} x_{c,\,2i+m,\,2j+n}
$$

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

```python
nn.MaxPool2d(2)
```

### 12. Global Average Pooling
- 長所: Linearのパラメータが激減し、サイズ合わせ不要
- 短所: 位置の情報が消える

$$
y_c = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c,i,j}
$$

```python
nn.AdaptiveAvgPool2d(output_size=1)
```

```python
nn.AdaptiveAvgPool2d(1)
```

## 正規化

### 13. Batch Normalization
- 長所: 学習が安定し速くなる
- 短所: バッチサイズが小さいと不安定

$$
\mathrm{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$

```python
nn.BatchNorm2d(num_features=24)
```

```python
nn.BatchNorm2d(channels)
```

### 14. Layer Normalization
- 長所: バッチサイズに左右されない
- 短所: CNNではBNより精度が出にくいことが多い

$$
\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

```python
nn.GroupNorm(num_groups=1, num_channels=24)
```

```python
nn.GroupNorm(1, channels)
```

## その他

### 15. Flatten + Linear
- 長所: 単純で分かりやすい
- 短所: 入力サイズに合わせる必要があり、パラメータが増えやすい

$$
y = W \, \mathrm{vec}(x) + b
$$

```python
nn.Flatten(start_dim=1)
nn.Linear(in_features=24, out_features=10)
```

```python
nn.Flatten(),
nn.Linear(channels * h * w, 10),
```

### 16. Dropout
- 長所: 過学習を抑える
- 短所: 学習が遅くなり、短い学習では逆効果のことも

$$
y = \frac{m \odot x}{1 - p}, \quad m_i \sim \mathrm{Bernoulli}(1 - p)
$$

```python
nn.Dropout(p=0.3)
```

## 最適化手法

### 17. SGD (Momentum)
- 長所: 最終的な精度が出やすい
- 短所: 学習率の調整がシビア

$$
v_t = \mu v_{t-1} + g_t, \quad \theta_t = \theta_{t-1} - \eta v_t
$$

```python
build_module_optimizer(name="sgd", model=model)
```

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```

### 18. Adam
- 長所: 調整なしでも速く収束する
- 短所: 最終精度はSGDに負けることがある

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

```python
build_module_optimizer(name="adam", model=model)
```

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 19. AdamW
- 長所: Adamより汎化しやすい
- 短所: weight_decayの調整が必要

$$
\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

```python
build_module_optimizer(name="adamw", model=model)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
```

### 20. Muon
- 長所: 少ないステップで速く学習できる
- 短所: 仕組みが難しく、optimizerを2つ使う必要がある

$$
M_t = \mu M_{t-1} + G_t, \quad \theta_t = \theta_{t-1} - \eta \, \mathrm{NewtonSchulz}(G_t + \mu M_t)
$$

```python
build_module_optimizer(name="muon", model=model)
```

```python
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"])
                g2d = g.reshape(g.size(0), -1)
                update = newton_schulz(g2d).view_as(p)
                scale = max(1, g2d.size(0) / g2d.size(1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)


muon_params  = [p for p in model.parameters() if p.ndim == 4]
adamw_params = [p for p in model.parameters() if p.ndim != 4]
optimizers   = [Muon(muon_params, lr=0.02), torch.optim.AdamW(adamw_params, lr=1e-3)]
```

## 学習率スケジューラ

`build_module_scheduler` に渡す名前で、学習率の変え方を選べます。$\eta_0$ はoptimizerの学習率、$e$ は今のepoch、$E$ は `train_epochs` です。

### 21. None
- 長所: 一番単純で、結果を比べやすい
- 短所: 最後まで学習率が大きく、精度が上がりきらないことがある

$$
\eta_e = \eta_0
$$

```python
schedulers = build_module_scheduler(name="none", optimizers=optimizers, epochs=train_epochs)
```

### 22. Cosine
- 長所: なめらかに下がり、調整なしでもよく効く
- 短所: 途中で学習を止めると、学習率が下がりきらない

$$
\eta_e = \frac{\eta_0}{2}\left(1 + \cos\frac{\pi e}{E}\right)
$$

```python
schedulers = build_module_scheduler(name="cosine", optimizers=optimizers, epochs=train_epochs)
```

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)
```

### 23. Step
- 長所: 仕組みが分かりやすい
- 短所: 学習率が急に変わり、下げる時期の調整が必要

$$
\eta_e = \eta_0 \cdot 0.1^{\lfloor e \,/\, \lfloor E/3 \rfloor \rfloor}
$$

```python
schedulers = build_module_scheduler(name="step", optimizers=optimizers, epochs=train_epochs)
```

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_epochs // 3, gamma=0.1)
```

### 24. WSD
- 長所: 大きい学習率で長く学習してから、最後に一気に仕上げられる（ここではWarmupはなし）
- 短所: 下げ始める時期（ここでは最後の20%）で結果が変わる

$$
\eta_e = \eta_0 \quad (e < 0.8E)
$$

$$
\eta_e = \eta_0 \cdot \frac{E - e}{E - 0.8E + 1} \quad (e \ge 0.8E)
$$

```python
schedulers = build_module_scheduler(name="wsd", optimizers=optimizers, epochs=train_epochs)
```

```python
decay_start = int(train_epochs * 0.8)
scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1.0 if e < decay_start else (train_epochs - e) / (train_epochs - decay_start + 1))
```

## 損失関数

### 25. Cross Entropy
- 長所: 分類の標準で安定している
- 短所: 自信過剰な予測になりやすい

$$
L = -\log \frac{e^{z_y}}{\sum_{j} e^{z_j}}
$$

```python
criterion = nn.CrossEntropyLoss()
```

### 26. Label Smoothing
- 長所: 1行で過学習を抑えられる
- 短所: εを大きくしすぎると精度が落ちる

$$
L = -\sum_{j} q_j \log p_j, \quad q_j = (1 - \varepsilon)\,\mathbb{1}[j = y] + \frac{\varepsilon}{K}
$$

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 27. Focal Loss
- 長所: 間違えやすい画像を重点的に学習できる
- 短所: 均等なデータ (CIFAR-10) ではあまり効かない

$$
L = -(1 - p_y)^{\gamma} \log p_y
$$

```python
class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, outputs, labels):
        ce  = nn.functional.cross_entropy(outputs, labels, reduction="none")
        p_y = torch.exp(-ce)
        return ((1 - p_y) ** self.gamma * ce).mean()


criterion = FocalLoss(gamma=2.0)
```

### 28. Multi-class Hinge Loss
- 長所: マージンを広げる考え方を学べる
- 短所: CEより精度が出にくいことが多い

$$
L = \frac{1}{K} \sum_{j \ne y} \max\left(0, \, 1 - z_y + z_j\right)
$$

```python
criterion = nn.MultiMarginLoss()
```
