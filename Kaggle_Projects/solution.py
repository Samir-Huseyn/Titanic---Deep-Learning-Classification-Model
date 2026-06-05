import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = sns.load_dataset("titanic")
df = df.to_csv("titanic.csv")
df = pd.read_csv("titanic.csv")


print(df.head(3))
print(df.isnull().sum())
print(df.info())
df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
df["age"] = df["age"].fillna(df["age"].median())
print(df["embarked"].value_counts())
df["embarked"] = df["embarked"].fillna("S")
df = df.drop(
    columns=[
        "deck",
        "id",
        "class",
        "who",
        "adult_male",
        "alive",
        "embarked",
        "embark_town",
        "parch",
        "sibsp",
        "alone",
    ]
)

df = pd.get_dummies(df, columns=["sex"], drop_first=True)
print(df.head(3))

X = df[['age', 'sex_male', 'pclass', 'fare']].values
y = df["survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model = Sequential([
    Dense(units=12, activation="relu", input_shape = (4,)),
    Dense(units=6, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs = 25, batch_size = 10)
loss, accuracy = model.evaluate(X_test, y_test)
print(loss, accuracy)


