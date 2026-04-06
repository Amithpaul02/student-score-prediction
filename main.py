import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'hours': [1,2,3,4,5,6,7,8],
    'scores': [10,20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

X = df[['hours']]
y = df['scores']

model = LinearRegression()
model.fit(X, y)

print("Prediction for 5 hours:", model.predict([[5]]))
