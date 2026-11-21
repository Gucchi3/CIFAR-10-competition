from datasets         import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision      import transforms
from torch.utils.flop_counter import FlopCounterMode

import os
import time
import torch
import torch.nn as nn

from module import ConvNeXtBlock, DepthwiseSeparableConv, ResidualBlock, SEBlock, SelfAttention, WindowAttention, build_module_optimizer

# 前処理済みデータの保存先
CACHE_PATH = "./cifar10_train.pt"

# 実行環境と前処理　※変更禁止
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

# Hugging FaceからCIFAR-10を読み込む　※変更禁止
if os.path.exists(CACHE_PATH):
    cache  = torch.load(CACHE_PATH)
    images = cache["images"]
    labels = cache["labels"]
else:
    hf_dataset = load_dataset("uoft-cs/cifar10", split="train")
    images     = torch.stack([transform(img) for img in hf_dataset["img"]])
    labels     = torch.tensor(hf_dataset["label"])
    torch.save({"images": images, "labels": labels}, CACHE_PATH + ".tmp")
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
dataset = TensorDataset(images, labels)

# 学習データをtrain 90%、valid 10%に分割　※変更禁止
train_size                   = int(len(dataset) * 0.9)
valid_size                   = len(dataset) - train_size
split_generator              = torch.Generator().manual_seed(0)
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=split_generator)

# loader　※変更禁止
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True )
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)






# model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2), # 32x32 -> 16x16

    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2), # 16x16 -> 8x8

    nn.Conv2d(32, 48, kernel_size=3, padding=1),
    nn.BatchNorm2d(48),
    nn.ReLU(),

    SelfAttention(48),

    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(48*1*1, 10),
).to(device)

# 学習設定
train_epochs = 30
criterion    = nn.CrossEntropyLoss()
optimizers   = build_module_optimizer("adam", model)








# 学習　※変更禁止
for epoch in range(train_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        for opt in optimizers: opt.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        for opt in optimizers: opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch {epoch + 1}: loss {avg_loss:.4f}")






# 最終モデルのvalid正答率を計算 ※変更禁止
model.eval()
correct = 0
total   = 0

with torch.inference_mode():
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs     = model(images)
        predictions = outputs.argmax(dim=1)
        correct    += (predictions == labels).sum().item()
        total      += labels.size(0)

num_params = sum(p.numel() for p in model.parameters())
print(f"Params: {num_params:,} / 40,000")
if num_params > 40_000:
    print("パラメータ数が上限を超えています")
x = torch.randn(1, 3, 32, 32, device=device)
with FlopCounterMode(display=False) as counter:
    model(x)
macs = counter.get_total_flops() // 2
print(f"MACs: {macs:,} / 6M")
accuracy = 100 * correct / total
if macs > 6000000:
    print("計算量が上限を超えています")
print(f"Valid accuracy: {accuracy:.2f}%")