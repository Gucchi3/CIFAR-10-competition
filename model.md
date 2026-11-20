# モデル
## 1.Self Attention
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


## 2.Residual Block
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