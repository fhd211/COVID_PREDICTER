import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


data= pd.read_csv('owid-covid-data.csv',sep=',')

data=data[['id','cases']]
print('-'*30);print('head');print('-'*30)

print(data.head())


print('-'*30);print('prepare data');print('-'*30)
x=np.array(data['id']).reshape(-1,1)
y=np.array(data['cases']).reshape(-1,1)

plt.plot(y,'-m')
plt.show()

polyFeat= PolynomialFeatures(degree=3)
x=polyFeat.fit_transform(x)
print(x)

print('-'*30);print('head');print('-'*30)
model= linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
print(f'Accuracy:{round(accuracy*100,3)}%')
y0=model.predict(x)
plt.plot(y0,'--b')
plt.show()



## prediction for upcomming days ##

days=2
print('-'*30);print('prediction');print('-'*30)
print(f'Prediction:{days} days:',end='')
print(round(int(model.predict(polyFeat.fit_transform([[234+days]])))/10000,2),'mill')


