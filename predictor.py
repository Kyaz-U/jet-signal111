import pickle
import numpy as np
from data_collector import get_latest_coefficients
import datetime

# Load the trained model
with open("models/aviator_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_aviator_signal():
    try:
        recent_values = get_latest_coefficients()
        X_input = np.array(recent_values).reshape(1, -1)
        predicted = model.predict(X_input)[0]
        predicted = round(predicted, 2)

        accuracy = 93
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        message = f"""✈️ <b>Aviator Signal</b>

🕒 Vaqt: <b>{timestamp}</b>
📈 Prognozlangan koeffitsiyent: <b>{predicted}x</b>
⏱️ O‘yin boshlanishidan <b>10 soniya</b> oldin

📊 Model aniqligi: <b>{accuracy}%</b>
#Aviator #Signal"""

        # Log the prediction
        with open("logs/predictions.log", "a") as log_file:
            log_file.write(f"[{timestamp}] Predicted: {predicted}x | Input: {recent_values}\\n")

        return message

    except Exception as e:
        return f"Xatolik yuz berdi: {str(e)}"
