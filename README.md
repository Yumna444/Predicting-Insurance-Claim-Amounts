🩺 Insurance Claim Amount Prediction

🎯 Objective:


Build a Linear Regression model to predict medical insurance charges based on personal and lifestyle factors.


📂 Dataset:


Medical Cost Personal Dataset named as insurance/ excel file


Each row represents a person, with features:


age: Age of the individual


sex: Gender (male/female)


bmi: Body Mass Index


children: Number of children


smoker: Whether the person smokes


region: Residential region in the US


charges: Insurance claim amount (target variable)


✅ Project Steps


1. Import Libraries


Used libraries for data processing, visualization, modeling, and evaluation:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error



2. Load & Explore Data

Loaded insurance.csv using pandas and explored the structure:


df = pd.read_csv("insurance.csv")
df.head()
df.info()



3. Data Preprocessing
🔹 Encode Categorical Features:
sex: male → 0, female → 1

smoker: no → 0, yes → 1

region: converted to dummy variables (one-hot encoding)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)



4. Data Visualization

Plotted key relationships to understand influence on charges:


Age vs Charges


BMI vs Charges


Smoker vs Charges


# Age vs Charges
sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
plt.title('Age vs Insurance Charges')
plt.show()
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/42a4f838-c860-4407-81af-dd925306871f" />


# BMI vs Charges
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('BMI vs Insurance Charges')
plt.show()
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/decabdca-40c3-4387-b051-734a9a409b46" />


# Boxplot: Smoker vs Charges
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Smoker vs Insurance Charges')
plt.xticks([0, 1], ['Non-Smoker', 'Smoker'])
plt.show()
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4cb9737f-af19-40cb-a31c-455a63f0cfe3" />


5. Feature & Target Definition

Split into inputs and output:


X = df.drop('charges', axis=1)
y = df['charges']



6. Train-Test Split & Model Training

Split the dataset and train a Linear Regression model:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)



7. Model Evaluation

Evaluate model using MAE and RMSE:


# Predict
y_pred = model.predict(X_test)


# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


📈 Conclusion

Smoking, age, and BMI significantly impact insurance charges.


Linear Regression gives a reasonable baseline model.


Model could be improved using more complex regressors (e.g., decision trees, ensemble models).

