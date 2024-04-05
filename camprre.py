import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
from torchsummary import summary

# 異常検知の閾値を設定
THRESHOLD_IMAGE = 0.4
THRESHOLD_PIXEL = 0.7

# モデルのパスを設定
model_path = 'padim_model.pth'


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Assuming you want to extract features from the last convolutional layer
        self.features = torch.nn.Sequential(*list(model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        print(f"Feature map shape: {x.size()}")  # This will print the shape of the feature map
        return x
# summary(model, (3, 224, 224))  # Adjust the input shape to your needs

def initialize_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.resnet50(pretrained=False)
    model = FeatureExtractor(base_model).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # チェックポイントから平均ベクトルと共分散行列の逆行列を読み込む
    mean_vector = checkpoint['mean']  # 'mean_vector' から 'mean' への修正
    inv_cov_matrix = checkpoint['cov_inv']  # こちらは修正不要

    # summary(model, (3, 224, 224), device=device.type)

    return model, mean_vector, inv_cov_matrix, device

# 異常検知のスコアを正規化する関数
def normalize_scores(scores):
    scores_min = np.min(scores)
    scores_max = np.max(scores)
    return (scores - scores_min) / (scores_max - scores_min)


def detect_anomalies(model, image, mean_vector, inv_cov_matrix, device):
    model.eval()
    with torch.no_grad():
        feature_maps = model(image)
        h, w = feature_maps.shape[-2], feature_maps.shape[-1]
        pixel_anomaly_scores = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                pixel_feature_vector = feature_maps[:, :, i, j].cpu().numpy()
                score = np.dot((pixel_feature_vector - mean_vector), inv_cov_matrix).dot((pixel_feature_vector - mean_vector).T)
                pixel_anomaly_scores[i, j] = np.sqrt(max(score, 0))  # Ensure non-negative

    # 正規化を行う前に異常スコアの最大値と最小値を計算します
    max_score = np.max(pixel_anomaly_scores)
    min_score = np.min(pixel_anomaly_scores)

    # スコアを0から1に正規化します
    normalized_scores = normalize_scores(pixel_anomaly_scores)
    return normalized_scores



def calculate_inv_cov_matrix(cov_matrix, regularization_term=1e-6):
    # 共分散行列の対角要素に小さな値を加える
    regularized_cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * regularization_term
    inv_cov_matrix = np.linalg.inv(regularized_cov_matrix)
    return inv_cov_matrix

def preprocess_image(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_image = preprocess(pil_image).unsqueeze(0)
    return processed_image

def generate_heatmap(normalized_scores, image_shape):
    # 0.4を超えるスコアに基づいてヒートマップを生成します
    thresholded_scores = np.where(normalized_scores > 0.4, normalized_scores, 0)
    heatmap = cv2.applyColorMap(np.uint8(255 * thresholded_scores), cv2.COLORMAP_JET)
    # ヒートマップを画像のサイズにリサイズします
    heatmap = cv2.resize(heatmap, (image_shape[1], image_shape[0]))
    return heatmap

def overlay_heatmap(frame, heatmap, alpha=0.5):
    # Overlay the heatmap onto the frame with transparency
    overlay_img = frame.copy()
    cv2.addWeighted(heatmap, alpha, overlay_img, 1 - alpha, 0, overlay_img)
    return overlay_img

def real_time_detection(model, mean_vector, inv_cov_matrix, device):
    cap = cv2.VideoCapture(0)  # カメラのキャプチャ開始
    while True:
        ret, frame = cap.read()
        if ret:
            processed_image = preprocess_image(frame).to(device)
            normalized_scores = detect_anomalies(
                model, processed_image, mean_vector, inv_cov_matrix, device
            )

            heatmap = generate_heatmap(normalized_scores, frame.shape[:2])
            overlay_frame = overlay_heatmap(frame, heatmap)

            # スコアのテキスト表示を更新
            text_score = f"Max: {normalized_scores.max():.2f}, Min: {normalized_scores.min():.2f}"
            cv2.putText(overlay_frame, text_score, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 画面表示
            cv2.imshow('Real-time Anomaly Detection with Heatmap', overlay_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model, mean_vector, inv_cov_matrix, device = initialize_model(model_path)
    real_time_detection(model, mean_vector, inv_cov_matrix, device)

