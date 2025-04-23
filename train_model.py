import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import os

# 1. CSV dan koeffitsiyentlarni o‘qish
df = pd.read_csv("aviator_real_data.csv")
coefficients = df["coefficient"].values

# 2. Model uchun tayyorlash (sliding window usuli)
window_size = 5
X, y = [], []
for i in range(len(coefficients) - window_size):
    X.append(coefficients[i:i+window_size])
    y.append(coefficients[i+window_size])

X = np.array(X)
y = np.array(y)

# 3. Modelni o‘qitish
model = LinearRegression()
model.fit(X, y)

# 4. Modelni saqlash
os.makedirs("modelss", exist_ok=True)
with open("modelss/aviator_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model muvaffaqiyatli o‘qitildi va saqlandi: modelss/aviator_model.pkl")
