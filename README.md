# CIFAR-10 コンペティション
## 内容
- CIFAR-10の学習コードを改良して、精度の高さを競います。
## 環境構築
次の4つのパッケージが必要です。numpy と Pillow は、これらと一緒に自動で入ります。
- torch
- torchvision
- datasets
- rich

```bash
pip install torch torchvision datasets rich
```
GPU で学習するときは、CUDA に合った torch を [PyTorch の公式サイト](https://pytorch.org/get-started/locally/) の手順で入れてください。  
動作を確認した環境: Python 3.14、torch 2.15、torchvision 0.29、datasets 5.2、rich 15.0
## 始め方
///
## 改良箇所
### モデル
AIにおいて、モデル構造は最も重要な要素といっても過言ではありません。  
先輩と一緒に、まずはここを改良しましょう。
### 学習設定
損失関数、学習率、オプティマイザ、スケジューラ、エポック数...とかいろいろあります。  
まずは、エポック数（epoch）や損失関数（criterion）、オプティマイザ（optimizer）からいじるのがおすすめ。
## 進め方  
まずは、先輩と一緒に**モデル**の改良をしてみましょう。  
分からないことがあれば、積極的に先輩に質問しましょう。


## ルール
- 検証データにおいて、一番精度の高いチームの勝ち。
- モデルには、畳み込み、アテンション、活性化関数、プーリングの枠組みの中から、それぞれ最低でも1つを用いること。
## 禁止事項
- 事前学習モデルの使用。
- 学習データの追加、分割周りの設定変更。
- seedの変更。（0固定）
- AI出力コードのコピペ。（積極的に先輩に質問しよう！）

## 変更一例
### モデル
```python
model = nn.Sequential(
    nn.Conv2d(3, 28, kernel_size=3, padding=1),
    nn.BatchNorm2d(28),
    nn.ReLU(),

    DConv2d(28, 28),
    nn.BatchNorm2d(28), 
    nn.ReLU(),
    nn.MaxPool2d(2), # 32x32 -> 16x16
    DConv2d(28, 56), 
    nn.BatchNorm2d(56), 
    nn.ReLU(),

    DConv2d(56, 56), 
    nn.BatchNorm2d(56), 
    nn.ReLU(),
    nn.MaxPool2d(2), # 16x16 -> 8x8
    DConv2d(56, 72), 
    nn.BatchNorm2d(72), 
    nn.ReLU(),
    DConv2d(72, 72), 
    nn.BatchNorm2d(72), 
    nn.ReLU(),

    SAttention(72),

    nn.AdaptiveAvgPool2d(1), 
    nn.Flatten(), 
    nn.Linear(72*1*1, 10),
).to(device)
```


### 学習設定
```python
train_epochs = 50
criterion    = nn.CrossEntropyLoss()
optimizers   = build_module_optimizer("muon", model)
schedulers   = build_module_scheduler("none", optimizers, train_epochs)
```

オプティマイザは `build_module_optimizer` に渡す名前で切り替わります。使える名前は `"adam"`、`"adamw"`、`"sgd"`、`"muon"` で、設定値は module.md と同じです。  
`"muon"` は、Convの重みをMuon（lr=0.02）で、それ以外（BN、Linear、bias）をAdamW（lr=1e-3）で更新します。そのため、Conv層が1つもないモデルでは使えません。

スケジューラは `build_module_scheduler` に渡す名前で切り替わります。使える名前は `"none"`（学習率を変えない）、`"cosine"`、`"step"`、`"wsd"` で、式は module.md にあります。学習率の大きさは optimizer ごとに決まっていて、スケジューラはそれを epoch ごとに下げていきます。
