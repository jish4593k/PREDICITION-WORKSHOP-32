import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tkinter as tk
from keras.models import Sequential
from keras.layers import Dense
import torch
from torch.autograd import Variable

# Load the dataset
df = pd.read_csv("pima-indians-diabetes.data.csv", header=None)
X = df.iloc[:, 0:8]
Y = df.iloc[:, 8]

# Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a neural network model using Keras
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=150, batch_size=10)

# Make predictions
Y_pred = model.predict(X_test)
Y_pred = [1 if y >= 0.5 else 0 for y in Y_pred]

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Data visualization with Seaborn
sns.pairplot(df, hue=8, diag_kind="kde")
plt.show()

# Create a simple GUI with Tkinter
def predict_diabetes():
    input_data = [float(entry.get()) for entry in input_entries]
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    result_label.config(text="Prediction: {}".format("Diabetic" if prediction[0][0] >= 0.5 else "Not Diabetic"))

root = tk.Tk()
root.title("Diabetes Predictor")

input_labels = ["Pregnancies:", "Glucose:", "BloodPressure:", "SkinThickness:", "Insulin:", "BMI:", "DiabetesPedigreeFunction:", "Age:"]
input_entries = []

for i, label_text in enumerate(input_labels):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    input_entries.append(entry)

predict_button = tk.Button(root, text="Predict", command=predict_diabetes)
predict_button.grid(row=len(input_labels), columnspan=2)

result_label = tk.Label(root, text="")
result_label.grid(row=len(input_labels) + 1, columnspan=2)

root.mainloop()
