# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ADARSH CHOWDARY R 
RegisterNumber: 212223040166 
```



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

# display the content in datafield
df.head()
df.tail()

#Segregating data to variables

X=df.iloc[:,:-1].values
X


Y=df.iloc[:,1].values
Y

#Splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

#displaying actual values
Y_test

#Graph plot for training data
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title('Hours vs Scores(Test Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse))
```

## Output:


Displaying the content in datafield

head:

![image](https://github.com/user-attachments/assets/8b8461bf-afa8-4420-b89c-036547528f6e)


tail:

![image](https://github.com/user-attachments/assets/5c6b382a-e9fe-425f-88f2-e766038de417)


Segregating data to variables:


![image](https://github.com/user-attachments/assets/fa0be54f-7113-4526-aef3-86a2eaa1b2fd)


![image](https://github.com/user-attachments/assets/03d8a26c-a4c1-4565-ba4e-9db4ccf9fec2)


Displaying predicted values:

![image](https://github.com/user-attachments/assets/cc45a2c9-430e-43b0-8847-665ddd6fe337)


Displaying actual values:

![image](https://github.com/user-attachments/assets/7c357c82-3ba8-42ab-a19a-34359d54ff09)


Graph plot for training data:

![image](https://github.com/user-attachments/assets/a83cf3cd-2957-486a-91ac-61515ec133ec)


Graph plot for test data:

![image](https://github.com/user-attachments/assets/d1ada528-01a4-485d-a4de-3a81cfb1248a)


MSE MAE RMSE:

![image](https://github.com/user-attachments/assets/92d4a908-8a48-4f55-a805-2947473150a7)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
