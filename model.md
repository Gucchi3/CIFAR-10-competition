# モデル

## 畳み込み系

### 1. Conv2d
$$
y_{c,i,j} = b_c + \sum_{k} \sum_{m,n} W_{c,k,m,n} \, x_{k,\,i+m,\,j+n}
$$

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
```

### 2. Residual Block
$$
y = \mathrm{ReLU}\left(x + \mathcal{F}(x)\right)
$$

$$
\mathcal{F}(x) = \mathrm{BN}\left(W_2 * \mathrm{ReLU}\left(\mathrm{BN}(W_1 * x)\right)\right)
$$

```python
class ResidualBlock(nn.Module):

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
$$
y = x + W_2 \, \mathrm{GELU}\left(W_1 \, \mathrm{Norm}\left(\mathrm{DWConv}_{7 \times 7}(x)\right)\right)
$$

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
$$
y = W_{\mathrm{pw}} * \mathrm{DWConv}_{3 \times 3}(x)
$$

```python
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

## Attention系

### 5. Self Attention
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

```python
class SelfAttention(nn.Module):

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
### 10. Window Attention
$$
y = x + \mathrm{merge}\left(\left\{\mathrm{softmax}\left(\frac{Q_w K_w^\top}{\sqrt{d}}\right)V_w\right\}_{w}\right)
$$

```python
class WindowAttention(nn.Module):

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
### 6. SE Block (チャネルattention)
$$
s = \sigma\left(W_2 \, \mathrm{ReLU}\left(W_1 \, \mathrm{GAP}(x)\right)\right), \quad y = s \odot x
$$

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

## 活性化関数

### 7. ReLU
$$
\mathrm{ReLU}(x) = \max(0, x)
$$

```python
nn.ReLU()
```

### 8. SiLU
$$
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

```python
nn.SiLU()
```

### 9. GELU
$$
\mathrm{GELU}(x) = x \cdot \Phi(x)
$$

```python
nn.GELU()
```

## プーリング

### 10. Max Pooling
$$
y_{c,i,j} = \max_{0 \le m,n < 2} x_{c,\,2i+m,\,2j+n}
$$

```python
nn.MaxPool2d(2)
```

### 11. Global Average Pooling
$$
y_c = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c,i,j}
$$

```python
nn.AdaptiveAvgPool2d(1)
```

## 正規化

### 12. Batch Normalization
$$
\mathrm{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$

```python
nn.BatchNorm2d(channels)
```

### 13. Layer Normalization
$$
\mathrm{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

```python
nn.GroupNorm(1, channels)
```

## その他

### 14. Flatten + Linear
$$
y = W \, \mathrm{vec}(x) + b
$$

```python
nn.Flatten(),
nn.Linear(channels * h * w, 10),
```

### 15. Dropout
$$
y = \frac{m \odot x}{1 - p}, \quad m_i \sim \mathrm{Bernoulli}(1 - p)
$$

```python
nn.Dropout(p=0.3)
```