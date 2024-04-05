import os
import torch
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from scipy.linalg import inv

# データ拡張と正規化の定義
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def collate_fn(batch):
    images = torch.stack([item for item in batch], dim=0)
    return images

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Feature extraction'):
            images = images.to(device)
            outputs = model(images)
            features_list.append(outputs.cpu().numpy())
    return np.concatenate(features_list, axis=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model = model.to(device)

    dataset = CustomDataset('./datasets/emboss/pass', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    features = extract_features(model, dataloader, device)
    features_flattened = features.reshape(features.shape[0], -1)
    mean_vector = np.mean(features_flattened, axis=0)
    cov_matrix = np.cov(features_flattened, rowvar=False)
    cov_inv = inv(cov_matrix)

    # モデルの状態辞書と統計プロファイルを保存
    torch.save({
        'model_state_dict': model.state_dict(),  # モデルの状態辞書を追加
        'mean': mean_vector,
        'cov': cov_matrix,
        'cov_inv': cov_inv  # 共分散行列の逆行列も保存
    }, 'padim_model.pth')

    print("Model and statistics saved.")

if __name__ == "__main__":
    main()
