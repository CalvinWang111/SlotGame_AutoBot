import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# 讀取數據
file_path = 'test_rtp.csv'
data = pd.read_csv(file_path)

class StateClassifier:
    def __init__(self, n_states):
        """
        初始化狀態分類器。
        :param n_states: 整數，表示要劃分的狀態數量。
        """
        self.n_states = n_states
        self.kmeans = KMeans(n_clusters=n_states, random_state=42)

    def fit(self, data):
        """
        使用 KMeans 對數據進行訓練。
        :param data: 二維數組，每行代表一個時間點，每列代表該時間點的物件值。
        """
        self.kmeans.fit(data)

    def predict(self, data):
        """
        根據數據預測每個時間點的狀態。
        :param data: 二維數組，每行代表一個時間點，每列代表該時間點的物件值。
        :return: 狀態列表，每個元素對應一個時間點的狀態或 'unknown'。
        """
        try:
            predictions = self.kmeans.predict(data)
        except ValueError:
            predictions = ["unknown"] * len(data)  # 如果有錯誤，則設定所有預測為 "unknown"
        return predictions

# 數據範例
example_data = data.iloc[:, 1:].values.T  # 忽略第一列 (Block)，並進行轉置

# 轉換數據為二元矩陣，NaN 或 NULL 視為 0 (未出現)，非零數值視為 1 (出現)
example_data_binary = np.where(np.isnan(example_data), 0, (example_data > 0).astype(int))

import seaborn as sns
import matplotlib.pyplot as plt

# 將二元數據轉換為 DataFrame
df_binary = pd.DataFrame(example_data_binary)

# 使用熱圖顯示資料
sns.heatmap(df_binary, cmap="YlGnBu", cbar=False)
plt.title("物件出現情況 (0: 未出現, 1: 出現)")
plt.show()
# 工具使用範例
n_states = 2  # 假設有 2 個狀態
classifier = StateClassifier(n_states)
classifier.fit(example_data_binary)

# 預測每個時間點的狀態
states = classifier.predict(example_data_binary)

# 輸出結果
print("狀態對應列表:", states)