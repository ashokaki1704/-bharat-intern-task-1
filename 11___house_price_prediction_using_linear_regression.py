import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt

csv_file_path = r'C:\Users\91939\Desktop\ml data sets\11_HousePricePredictionusing_LinearRegression\dataset.csv'

dataset = pd.read_csv(csv_file_path)

print(dataset.shape)
print(dataset.head(5))

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(dataset.area,dataset.price,color='red',marker='*')

X = dataset.drop('price',axis='columns')
X
Y = dataset.price
Y

model = LinearRegression()
model.fit(X,Y)

x=float(input("enter a prediction land area in sqft : "))
LandAreainSqFt=[[x]]
PredictedmodelResult = model.predict(LandAreainSqFt)
print(PredictedmodelResult)

m=model.coef_
print(m)

b=model.intercept_
print(b)

y = m*x + b
print("The Price of {0} Square feet Land is: {1}".format(x,y[0]))