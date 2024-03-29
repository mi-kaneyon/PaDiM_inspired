import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

# データセットクラス（ラベルを含む）
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 良品の画像を読み込み
        pass_dir = os.path.join(root_dir, 'pass')
        for img_name in os.listdir(pass_dir):
            self.images.append(os.path.join(pass_dir, img_name))
            self.labels.append(0)  # 良品は0

        # 不良品の画像を読み込み
        defect_dir = os.path.join(root_dir, 'defect')
        for img_name in os.listdir(defect_dir):
            self.images.append(os.path.join(defect_dir, img_name))
            self.labels.append(1)  # 不良品は1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # CustomDatasetインスタンスを作成
    dataset = CustomDataset(root_dir='datasets/emboss', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ResNet50モデルのインスタンス化
    model = models.resnet50(weights=None)  # ここではweights=Noneを指定しています

    # 最終層をタスクに合わせて調整（ここでは2クラス分類を想定）
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    # 学習済みのモデルの重みをロード
    model.load_state_dict(torch.load('model.pth', map_location=device))

    # デバイスにモデルを移動
    model.to(device)

    # バリデーションを実行
    validate(model, dataloader, device)


if __name__ == "__main__":
    main()
