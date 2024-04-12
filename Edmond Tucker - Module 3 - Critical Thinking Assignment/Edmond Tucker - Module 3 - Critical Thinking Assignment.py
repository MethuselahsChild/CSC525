import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
csv_file_path = os.path.join(script_dir, 'Salary_Data.csv')

df = pd.read_csv(csv_file_path)

X = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X_train)

model = LinearRegression()
model.fit(X_poly, y_train)

X_test_poly = poly_features.transform(X_test)
y_pred = model.predict(X_test_poly)

print("MSE: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance Score: %.2f' % r2_score(y_test, y_pred))

def predict_salary(years_of_experience):
    years = np.array(years_of_experience).reshape(-1, 1)
    years_poly = poly_features.transform(years)
    salary_pred = model.predict(years_poly)
    return salary_pred[0]

experience = input("Candidate's Years of Experience: ")
experience = float(experience)
predicted_salary = predict_salary(experience)
print(f"Predicted Salary for {experience} years of experience: {predicted_salary}")

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(poly_features.fit_transform(X)), color='green')
plt.title('Employee Salary Prediction (Polynomial Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Employee Salary')
plt.show()
