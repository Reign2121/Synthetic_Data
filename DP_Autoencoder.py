import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# 전처리 함수
def preprocess_data(df, categorical_features, numerical_features):
    onehot_encoders = {}
    scaler = MinMaxScaler()
    encoded_columns = []

    # 명목형 변수 원-핫 인코딩
    for column in categorical_features:
        onehot_encoders[column] = OneHotEncoder(dtype='int8', sparse_output=False)#, drop="first")
        # Series를 DataFrame으로 변환하여 fit_transform 수행
        encoded_col = onehot_encoders[column].fit_transform(df[[column]])
        encoded_col_df = pd.DataFrame(encoded_col, columns=onehot_encoders[column].get_feature_names_out([column]))
        encoded_columns.append(encoded_col_df)
    
    # 원-핫 인코딩된 컬럼들을 데이터프레임으로 결합
    df_encoded = pd.concat(encoded_columns, axis=1)

    # 원래의 수치형 변수들을 데이터프레임에 추가
    df_encoded = pd.concat([df_encoded, df[numerical_features].reset_index(drop=True)], axis=1)

    # 모든 변수들 Min-Max 스케일링
    df_scaled = scaler.fit_transform(df_encoded)
    input_dim = df_scaled.shape[1]
        
    return df_scaled, onehot_encoders, scaler, input_dim



# 데이터 로더 생성 함수
def create_data_loader(df_scaled, batch_size):
    df_tensor = torch.tensor(df_scaled, dtype=torch.float32)
    print(df_tensor.shape)
    dataset = TensorDataset(df_tensor, df_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, df_tensor


def inverse_transform(df_anonymized, onehot_encoders, scaler, categorical_features, numerical_features, original_columns):
    # Min-Max 스케일링 역변환
    df_restored = scaler.inverse_transform(df_anonymized)

    # 복원된 데이터를 저장할 딕셔너리
    restored_data = {}
    current_idx = 0

    # 명목형 변수 복원
    for column in categorical_features:
        # 원-핫 인코딩된 열의 개수를 계산
        num_encoded_cols = len(onehot_encoders[column].categories_[0])
        
        # 인코딩된 열들을 가져오기
        encoded_cols = df_restored[:, current_idx:current_idx + num_encoded_cols]

        # 원-핫 인코딩된 데이터의 역변환
        decoded_col = onehot_encoders[column].inverse_transform(encoded_cols)

        # 복원된 범주형 변수를 1차원 배열로 변환하여 딕셔너리에 추가
        restored_data[column] = decoded_col.flatten()
        
        # 인덱스 업데이트
        current_idx += num_encoded_cols

    # 수치형 변수 복원
    for column in numerical_features:
        restored_data[column] = df_restored[:, current_idx]
        current_idx += 1

    # 복원된 모든 열을 결합하여 데이터프레임 생성
    df_restored = pd.DataFrame(restored_data)
    
    # 원래 데이터 프레임의 변수 순서대로 정렬
    df_restored = df_restored[original_columns]

    return df_restored




import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine #pip install opacus

# GPU 사용 가능 확인 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Swish 활성화 함수 정의
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Autoencoder 클래스를 전역으로 정의
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.swish = Swish()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, (input_dim // 2)//2),
            #Swish(),
            #nn.Dropout(0.3),
            #nn.Linear(64, 32),
            #Swish(),
        )
        
        self.decoder = nn.Sequential(
            #nn.Linear(32, 64),
            #Swish(),
            #nn.Dropout(0.3),
            nn.Linear((input_dim // 2)//2, (input_dim // 2)),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(input_dim//2, input_dim),
            #nn.Sigmoid()  # assuming data is in [0, 1] range
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# AnonymizedAutoencoder 클래스 정의
class DP_Autoencoder:
    def __init__(self, input_dim, num_epochs=50, batch_size=16, noise_multiplier=1.0, max_grad_norm=10.0, learning_rate=0.0001):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate

        # 모델 생성
        self.model = Autoencoder(input_dim).to(device)  # 모델을 GPU로 이동

    def train_model(self, data_loader):
        # 옵티마이저 정의
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # PrivacyEngine 초기화 및 모델/옵티마이저/데이터로더에 통합
        privacy_engine = PrivacyEngine()

        self.model, optimizer, data_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        # 손실 함수 정의
        criterion = nn.MSELoss()

        # 모델 학습
        for epoch in range(self.num_epochs):
            for inputs, _ in data_loader:
                inputs = inputs.to(device)  # 데이터를 GPU로 이동
                # 순전파
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 프라이버시 손실 계산
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Reconstruction Error(Loss): {loss.item():.4f}, (ε = {epsilon:.2f}, δ = 1e-5)")

    def get_new_data(self, df_tensor):
        df_tensor = df_tensor.to(device)  # 데이터를 GPU로 이동
        # 익명화된 특성 추출
        with torch.no_grad():
            df_anonymized = self.model(df_tensor).cpu().numpy()  # 결과를 CPU로 이동
        print(df_anonymized.shape)
        return df_anonymized 
