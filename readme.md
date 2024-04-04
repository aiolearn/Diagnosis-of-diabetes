<h1 align="center">Welcome to Diagnosis of diabetes Project 👋</h1>

# Diagnosis of diabetes

In this project, we have written a diabetes diagnosis program with regression and Python

## Modules

```python
import math
import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
```

## Usage

Reading data from CSV file and displaying its 5 lines

```python
df = pd.read_csv('diabetes.csv')
df.head()
df.describe()
```

Display the number of results for each value of 0 and 1

```python
sns.countplot(x='Outcome', data=df)
print('تعداد نتایج برای هر مقدار 0 و 1:\n',
      df['Outcome'].value_counts())
```

A heat map display of data correlation

```python
plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True)
```

Split data into attributes and tags

```python
x = df.drop("Outcome", axis=1)
y = df.Outcome
y.head()
y.shape
```

Divide the data into training and test data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
Enter the logistic regression model and build

```python
from sklearn import linear_model
model=linear_model.LogisticRegression()
model.fit(X_train,y_train)
```

Building a neural network model

```python
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
```

Compile the model

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Training the model with training data

```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=10)
```

Evaluation of model accuracy on training and test data

```python
scores = model.evaluate(X_train, y_train)
print("دقت آموزش: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test)
print("دقت تست: %.2f%%\n" % (scores[1]*100))
```

Predict labels for test data

```python
y_test_pred = model.predict(X_test)
```

Calculate the number of correct and incorrect predictions

```python
itr = 0
ifa = 0
for ii, i in enumerate(y_test):
    xp = -1
    tf = ""

    if y_test_pred[ii] >= 0.7:
        xp = 1
    else:
        xp = 0

    if i == xp:
        tf = ""
        itr = itr+1
    else:
        tf = "false"
        ifa = ifa+1
```

Display the number of correct and incorrect predictions

```python
print("صحیح:", itr, "  غلط:", ifa)
```

Calculate the percentage of accuracy

```python
z = itr+ifa
p = (itr*100)/z
print(p)
```

Forecast for a new sample

```python
man = np.array([[0, 100, 100, 50, 100, 32, 0.2, 56]])
out1 = model.predict(man)
print(out1)
```

Show details of model layers

```python
for layer in model.layers:
    print("Layer name: " + layer.name)
    print("Layer type: " + layer.__class__.__name__)
    print("Input dimensions: {}".format(layer.input_shape[1:]))
    print("Output dimensions: {}".format(layer.output_shape[1:]))
    print("Number of parameters: {}".format(layer.count_params()))
    try:
        print("Activation function: " + layer.activation.__name__)
        print(" ")
    except:
        print(" ")
```

Save model weights

```python
model.save_weights("model.h5")
```

Save model architecture

```python
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
```

Load model weights

```python
model.load_weights('model.h5')
```

## Result

This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️


# تشخیص دیابت

در این پروژه برنامه تشخیص دیابت با رگرسیون و پایتون نوشته ایم

## ماژول ها

```python
import math
import numpy as np
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
```

## نحوه استفاده

خواندن داده ها از فایل CSV و نمایش 5 خط آن

```python
df = pd.read_csv('diabetes.csv')
df.head()
df.describe()
```

نمایش تعداد نتایج برای هر مقدار 0 و 1

```python
sns.countplot(x='Outcome', data=df)
print('تعداد نتایج برای هر مقدار 0 و 1:\n',
      df['Outcome'].value_counts())
```

نمایش نقشه حرارتی از همبستگی داده ها

```python
plt.subplots(figsize=(9, 9))
sns.heatmap(df.corr(), annot=True)
```

داده ها را به ویژگی ها و برچسب ها تقسیم کنید

```python
x = df.drop("Outcome", axis=1)
y = df.Outcome
y.head()
y.shape
```

داده ها را به داده های آموزشی و آزمایشی تقسیم کنید

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
مدل رگرسیون لجستیک را وارد کرده و بسازید

```python
from sklearn import linear_model
model=linear_model.LogisticRegression()
model.fit(X_train,y_train)
```

ساخت مدل شبکه عصبی

```python
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
```

مدل را کامپایل کنید

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

آموزش مدل با داده های آموزشی

```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=10)
```

ارزیابی دقت مدل در آموزش و داده های آزمون

```python
scores = model.evaluate(X_train, y_train)
print("دقت آموزش: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test)
print("دقت تست: %.2f%%\n" % (scores[1]*100))
```

برچسب‌ها را برای داده‌های آزمایشی پیش‌بینی کنید

```python
y_test_pred = model.predict(X_test)
```

تعداد پیش بینی های صحیح و نادرست را محاسبه کنید

```python
itr = 0
ifa = 0
for ii, i in enumerate(y_test):
    xp = -1
    tf = ""

    if y_test_pred[ii] >= 0.7:
        xp = 1
    else:
        xp = 0

    if i == xp:
        tf = ""
        itr = itr+1
    else:
        tf = "false"
        ifa = ifa+1
```

نمایش تعداد پیش بینی های صحیح و نادرست

```python
print("صحیح:", itr, "  غلط:", ifa)
```

درصد دقت را محاسبه کنید

```python
z = itr+ifa
p = (itr*100)/z
print(p)
```

پیش بینی برای نمونه جدید

```python
man = np.array([[0, 100, 100, 50, 100, 32, 0.2, 56]])
out1 = model.predict(man)
print(out1)
```

نمایش جزئیات لایه های مدل

```python
for layer in model.layers:
    print("Layer name: " + layer.name)
    print("Layer type: " + layer.__class__.__name__)
    print("Input dimensions: {}".format(layer.input_shape[1:]))
    print("Output dimensions: {}".format(layer.output_shape[1:]))
    print("Number of parameters: {}".format(layer.count_params()))
    try:
        print("Activation function: " + layer.activation.__name__)
        print(" ")
    except:
        print(" ")
```

وزن مدل را ذخیره کنید

```python
model.save_weights("model.h5")
```

ذخیره معماری مدل

```python
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
```

وزن مدل را بارگیری کنید

```python
model.load_weights('model.h5')
```

## نتیجه

این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️