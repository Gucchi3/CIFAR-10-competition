from datasets         import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision      import transforms

import os
import time
import torch
import torch.nn as nn

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

    nn.Flatten(),
    nn.Linear( *  * , 10),
).to(device)

# 学習設定
train_epochs = 30
criterion    = nn.CrossEntropyLoss()
optimizer    = torch.optim.Adam(model.parameters())

# 学習
for epoch in range(train_epochs):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

accuracy = 100 * correct / total
print(f"Valid accuracy: {accuracy:.2f}%")