# 🚢 Titanic - Deep Learning Classification Model

This project utilizes the famous **Titanic** historical passenger dataset to build an **Artificial Neural Network (ANN)** that predicts whether a passenger would survive the shipwreck.

The primary goal of this project is to move away from synthetic data and apply real-world Data Engineering and Deep Learning principles to handle noisy, historical data.

---

## 🛠️ Tech Stack & Libraries
* **Python 3.11+**
* **Pandas & NumPy** — For data manipulation and engineering
* **Seaborn** — For loading the real-world dataset
* **Scikit-Learn** — For Data Splitting and Feature Scaling (`StandardScaler`)
* **TensorFlow & Keras** — For designing and training the Neural Network

---

## 📊 Data Preprocessing
To ensure the neural network trains efficiently without mathematical bias, the following real-world data engineering steps were taken:
1. **Handling Missing Values:** Missing entries in the `age` column were filled using the dataset's median age, and missing `embarked` values were imputed with the most frequent port ("S").
2. **Feature Dropping:** Unnecessary, redundant, or leaky columns that could confuse the model (`deck`, `class`, `who`, `alive`, etc.) were dropped.
3. **Categorical Encoding:** The `sex` column was converted into a binary numerical format using `pd.get_dummies` (resulting in a `sex_male` column of 0s and 1s).
4. **Feature Scaling:** `StandardScaler` was applied to normalize numerical variations (`age`, `fare`), ensuring that Gradient Descent converges rapidly without zig-zagging.

---

## 🧠 Neural Network Architecture
The model is built using the **Keras Sequential API** with a 3-layer architecture:

* **Input Layer:** Accepts 4 features (`age`, `sex_male`, `pclass`, `fare`) mapping into **12 neurons** with a `ReLU` activation function.
* **Hidden Layer:** Consists of **6 neurons** with a `ReLU` activation function to capture non-linear relationships.
* **Output Layer:** Comprises **1 neuron** with a `Sigmoid` activation function to output a probability between 0 and 1 (Survived vs. Did Not Survive).

```python
model = Sequential([
    Dense(units=12, activation="relu", input_shape=(4,)),
    Dense(units=6, activation="relu"),
    Dense(units=1, activation="sigmoid")
])
