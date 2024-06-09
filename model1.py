import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle



df = pd.read_csv('kc_house_data.csv')

# Exploratory Data Analysis
zipcode=df.iloc[:,16]
uz=pd.unique(zipcode)
# calculate price per square footage of living for each zipcode
# and store in zipcode_up
zipcode_up = pd.Series([])         
            
for i in range(len(uz)):
    count = 0
    temp_sum =0
    for k in range(len(df)):
        if (df.iloc[:,16][k] == uz[i]):
            temp_sum =temp_sum + df.iloc[:,2][k]/df.iloc[:,5][k]
            count = count + 1
    zipcode_up[i]=temp_sum/count


# insert zipcode_up column into df
zipcode_up_column = pd.Series([]) 
for i in range(len(uz)):
    for k in range(len(df)):
        if (df.iloc[:,16][k] == uz[i]):
            zipcode_up_column[k] = zipcode_up[i]

df.insert(17, "zipcode_up", zipcode_up_column)
df=df.drop(['date'],axis=1)
df= df.astype(float)
     


# Partitioning a dataset into separate
# training and test sets


# Using the fit method, StandardScaler estimated the
# parameters μ (sample mean) and σ (standard deviation) for each feature dimension
# from the training data. By calling the transform method, we then standardized the
# training data using those estimated parameters μ and σ. Note that we used the
# same scaling parameters to standardize the test set so that both the values in the
# training and test dataset are comparable to each other.



X = df[['grade','sqft_living','zipcode_up']].values
y = df['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc_x = StandardScaler()
sc_y = StandardScaler()

X_train_std = sc_x.fit_transform(X_train)
X_test_std = sc_x.transform(X_test)

y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
y_test_std = sc_y.transform(y_test[:, np.newaxis]).flatten()

# polynomial regression

lrg = LinearRegression()

# create quadratic_retail features
cubic = PolynomialFeatures(degree=3)

X_cubic_train = cubic.fit_transform(X_train_std)
X_cubic_test = cubic.transform(X_test_std)


#Fitting model with trainig data
lrg.fit(X_cubic_train, y_train_std)

# Saving model to disk
pickle.dump(lrg, open('model1.pkl','wb'))

# Loading model to compare the results
retailmodel = pickle.load(open('model1.pkl','rb'))

print("Price $ %.3f" % sc_y.inverse_transform(
    retailmodel.predict(cubic.transform(
        sc_x.transform(np.array([7,1960,352.677904]).reshape(1, -1))))))


# Saving functions to disk

pickle.dump(sc_x, open('sc_x.pkl','wb'))
pickle.dump(sc_y, open('sc_y.pkl','wb'))
pickle.dump(cubic, open('cubic.pkl','wb'))


