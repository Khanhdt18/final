import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.ensemble import GradientBoostingRegressor



df = pd.read_csv('kc_house_data.csv')

# just take the year from the date column
df['sales_yr']=df['date'].astype(str).str[:4]

# add the age of the buildings when the houses were sold as a new column
df['age']=df['sales_yr'].astype(int)-df['yr_built']
# add the age of the renovation when the houses were sold as a new column
df['age_rnv']=0
df['age_rnv']=df['sales_yr'][df['yr_renovated']!=0].astype(int)-df['yr_renovated'][df['yr_renovated']!=0]
df['age_rnv'][df['age_rnv'].isnull()]=0

for i in range(len(df)):
    if (df['age_rnv'][i]== 0):
        df['age_rnv'][i]=df['age'][i]




df.rename(columns={'date':'sales_time'}, inplace=True)
year=['2014','2015']
month=[0,1,2,3,4,5,6,7,8,9,10,11,12]
for i in range(len(df)):
    if df['sales_yr'][i]==year[1] and int(df.iloc[:,1][i][4:6])==5:
        df.iloc[:,1][i]= month[0]
    elif df['sales_yr'][i]==year[1] and int(df.iloc[:,1][i][4:6])==4:
        df.iloc[:,1][i]= month[1]
    elif df['sales_yr'][i]==year[1] and int(df.iloc[:,1][i][4:6])==3:
        df.iloc[:,1][i]= month[2]
    elif df['sales_yr'][i]==year[1] and int(df.iloc[:,1][i][4:6])==2:
        df.iloc[:,1][i]= month[3]
    elif df['sales_yr'][i]==year[1] and int(df.iloc[:,1][i][4:6])==1:
        df.iloc[:,1][i]= month[4]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==12:
        df.iloc[:,1][i]= month[5]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==11:
        df.iloc[:,1][i]= month[6]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==10:
        df.iloc[:,1][i]= month[7]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==9:
        df.iloc[:,1][i]= month[8]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==8:
        df.iloc[:,1][i]= month[9]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==7:
        df.iloc[:,1][i]= month[10]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==6:
        df.iloc[:,1][i]= month[11]
    elif df['sales_yr'][i]==year[0] and int(df.iloc[:,1][i][4:6])==5:
        df.iloc[:,1][i]= month[12]
  
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
df= df.astype(float)
     


# make use of momentum effect
sales_time_up = pd.Series([]) 

for i in range(len(month)):
    count = 0
    temp_sum =0
    for k in range(len(df)):
        if (df.iloc[:,1][k] == month[i]):
            temp_sum =temp_sum + df.iloc[:,2][k]/df.iloc[:,5][k]
            count = count + 1
    sales_time_up[i]=temp_sum/count

# insert zipcode_up column into df
sales_time_up_column = pd.Series([])
for i in range(len(month)):
    for k in range(len(df)):
        if (df.iloc[:,1][k] == month[i]):
            sales_time_up_column[k] = sales_time_up[i]

df.insert(3, "sales_time_up", sales_time_up_column)
df= df.astype(float)
          


# Using the fit method, StandardScaler estimated the
# parameters μ (sample mean) and σ (standard deviation) for each feature dimension
# from the training data. By calling the transform method, we then standardized the
# training data using those estimated parameters μ and σ. Note that we used the
# same scaling parameters to standardize the test set so that both the values in the
# training and test dataset are comparable to each other.


        



# Here we add model 2 for corporate customers
P = df[['sales_time_up', 'bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade','sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15', 'age', 'age_rnv','long','lat','zipcode_up']].values
q = df['price'].values
P_train, P_test, q_train, q_test = train_test_split(P, q, test_size=0.3, random_state=0)

tx = StandardScaler()
ty = StandardScaler()

P_train_std = tx.fit_transform(P_train)
P_test_std = tx.transform(P_test)

q_train_std = ty.fit_transform(q_train[:, np.newaxis]).flatten()
q_test_std = ty.transform(q_test[:, np.newaxis]).flatten()

# GradientBoostingRegressor

GBRegr = GradientBoostingRegressor(random_state=0, max_depth =3, n_estimators=1000)

#Fitting model with trainig data
GBRegr.fit(P_train_std, q_train_std)

# Saving model to disk
pickle.dump(GBRegr, open('model2.pkl','wb'))

# Loading model to compare the results
corpmodel = pickle.load(open('model2.pkl','rb'))

print("Price $ %.3f" % ty.inverse_transform(corpmodel.predict(tx.transform(np.array([284.410837,4.0,3.00,1960.0,5000.0,1.0,0.0,0.0,5.0,7.0,1050, 910, 1360, 5000 ,49.0,49.0,-122.045,47.5208,337.218034]).reshape(1, -1)))))

# record functions to disk

pickle.dump(tx, open('tx.pkl','wb'))
pickle.dump(ty, open('ty.pkl','wb'))
pickle.dump(uz, open('uz.pkl','wb'))
pickle.dump(GBRegr, open('GBRegr.pkl','wb'))
pickle.dump(sales_time_up, open('sales_time_up.pkl','wb'))
pickle.dump(zipcode_up, open('zipcode_up.pkl','wb'))
