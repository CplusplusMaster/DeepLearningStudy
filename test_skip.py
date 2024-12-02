import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix
from torchsummary import summary

# Skip Connection이 포함된 AlexNet Feature Extractor
class AlexNetWithSkip(nn.Module):
    def __init__(self, pretrained_model):
        super(AlexNetWithSkip, self).__init__()
        self.features = nn.Sequential(*list(pretrained_model.features.children())[:])
        self.skip = nn.Conv2d(256, 256, kernel_size=1)  # Skip Connection Conv layer
        self.pool = nn.AdaptiveAvgPool2d((6, 6))  # Pooling layer to match dimensions

    def forward(self, x):
        skip_output = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 8:  # Add skip connection at Conv5 (Layer Index 8)
                skip_output = self.pool(x)
        x = x + skip_output  # Skip Connection Added
        return x

# Linear Regressor 연결
class LinearRegressor(nn.Module):
    def __init__(self, feature_dim, output_dim=10):
        super(LinearRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        return x

# # PCA를 위한 함수
# def perform_pca(features, n_components=2):
#     pca = PCA(n_components=n_components)
#     reduced_data = pca.fit_transform(features)
#     return reduced_data

# Feature 추출 함수
def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def main():
    # 데이터 전처리 및 로드
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    
    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    # Pretrained AlexNet 가져오기
    alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    # 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AlexNetWithSkip(alexnet).to(device)
    regressor = LinearRegressor(feature_dim=9216, output_dim=10).to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)

    # 모델 요약 출력 
    print("--Feature Extractor--")
    summary(feature_extractor, input_size=(3, 224, 224), device=str(device))

    print("--Regressor--")
    summary(regressor, input_size=(9216,), device=str(device))


    # Feature Extractor 고정
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # 학습 루프
    def train_model(feature_extractor, regressor, trainloader, epochs=100):
        feature_extractor.eval()  # Feature Extractor 고정
        for epoch in range(epochs):
            regressor.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.no_grad():
                    features = feature_extractor(inputs)  # Feature Extractor 통과
                outputs = regressor(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

    train_model(feature_extractor, regressor, trainloader, epochs=100)

    # 성능 평가
    def evaluate_model(feature_extractor, regressor, testloader):
        feature_extractor.eval()
        regressor.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = feature_extractor(inputs)
                outputs = regressor(features)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Top-1, Top-3 Accuracy 및 Confusion Matrix
        top1_accuracy = accuracy_score(all_labels, np.argmax(all_preds, axis=1))
        top3_accuracy = top_k_accuracy_score(all_labels, all_preds, k=3, labels=np.arange(len(classes)))
        conf_matrix = confusion_matrix(all_labels, np.argmax(all_preds, axis=1))

        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    evaluate_model(feature_extractor, regressor, testloader)

    # # PCA 구현 함수 2d
    # def perform_pca(features, n_components=2):
    #     # Data Centering
    #     mean = np.mean(features, axis=0)
    #     centered_data = features - mean
    #     # Covariance Matrix 계산
    #     covariance_matrix = np.cov(centered_data, rowvar=False)
    #     # Eigenvalues, Eigenvectors 계산
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #     # Principal Components 선택
    #     sorted_indices = np.argsort(eigenvalues)[::-1]
    #     top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    #     # Embedding Space로 변환
    #     reduced_data = np.dot(centered_data, top_eigenvectors)
    #     return reduced_data, top_eigenvectors
    # # Feature 추출 및 PCA 수행
    # features, labels = extract_features(feature_extractor, testloader, device)
    # reduced_data, _ = perform_pca(features, n_components=2)

    # # PCA 결과 시각화
    # plt.figure(figsize=(10, 8))
    # for i, class_name in enumerate(classes):
    #     idx = labels == i
    #     plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=class_name, alpha=0.5)
    # plt.title("PCA Embedding Space")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.legend()
    # plt.show()

    from mpl_toolkits.mplot3d import Axes3D

    # PCA 구현 함수 (3D 확장 가능)
    def perform_pca(features, n_components=3):
        # Data Centering
        mean = np.mean(features, axis=0)
        centered_data = features - mean

        # Covariance Matrix 계산
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Eigenvalues, Eigenvectors 계산
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Principal Components 선택
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

        # Embedding Space로 변환
        reduced_data = np.dot(centered_data, top_eigenvectors)
        return reduced_data, top_eigenvectors

    # Feature 추출 및 PCA 수행
    features, labels = extract_features(feature_extractor, testloader, device)
    reduced_data, _ = perform_pca(features, n_components=3)

    # 3D PCA 결과 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, class_name in enumerate(classes):
        idx = labels == i
        ax.scatter(reduced_data[idx, 0], reduced_data[idx, 1], reduced_data[idx, 2], label=class_name, alpha=0.5)

    ax.set_title("3D PCA Embedding Space")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
