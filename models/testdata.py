from scipy.io import arff
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import xlrd
from sklearn.model_selection import train_test_split

data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv('../data_regression/bike_sharing_hour.csv')
print(rides.head(2))
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=True)
    rides = pd.concat([rides, dummies], axis=1)
print(rides.head(2))
fields_to_drop = ['instant', 'dteday', 'season', 'atemp', 'yr', 'registered', 'casual', 'season',
                  'weathersit', 'mnth', 'hr', 'weekday'] #remove original features
data = rides.drop(fields_to_drop, axis=1)
print(data.head(2))
data = pd.DataFrame(data)
X = data.iloc[:, 0:51].values
y = data.iloc[:, 51].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize logistic regression model
lModel = LinearRegression()

# Train the model
lModel.fit(X_train, y_train)
lModel.predict(X_test)
print(lModel.score(X_test, y_test))



# data = pd.read_excel('../data/default_of_credit_card_clients.xls')
# # print(data.head(20))
# dataset=pd.DataFrame(data)
# dataset=dataset.iloc[1:,1:]
# print(dataset)
# print(dataset.iloc[:,0:23].values.astype(int))
# print(dataset.iloc[:,23].values).astype(int)




# data = arff.loadarff('../data/messidor_features.arff')
# dataset = pd.DataFrame(data[0])
# print(dataset)
# X = dataset.iloc[:, 0: 19].values
# y = dataset.iloc[:, 19].values.astype('int') # convert to int from object type
# print(X)

# data = arff.loadarff('../data/seismic-bumps.arff')
# dataset = pd.DataFrame(data[0])
# label_encoder = LabelEncoder()
# dataset['seismic'] = label_encoder.fit_transform(dataset['seismic'])
# dataset['seismoacoustic'] = label_encoder.fit_transform(dataset['seismoacoustic'])
# dataset['shift'] = label_encoder.fit_transform(dataset['shift'])
# dataset['ghazard'] = label_encoder.fit_transform(dataset['ghazard'])
# dataset['class'] = label_encoder.fit_transform(dataset['class'])
# print(dataset)
# X = dataset.iloc[:, 0: 18].values.astype(int)
# y = dataset.iloc[:, 18].values
# print(X)
# print(y)


# data = arff.loadarff('../data/ThoraricSurgery.arff')
# dataset = pd.DataFrame(data[0])
# label_encoder = LabelEncoder()
# dataset['DGN'] = label_encoder.fit_transform(dataset['DGN'])
# dataset['PRE6'] = label_encoder.fit_transform(dataset['PRE6'])
# dataset['PRE7'] = label_encoder.fit_transform(dataset['PRE7'])
# dataset['PRE8'] = label_encoder.fit_transform(dataset['PRE8'])
# dataset['PRE9'] = label_encoder.fit_transform(dataset['PRE9'])
# dataset['PRE10'] = label_encoder.fit_transform(dataset['PRE10'])
# dataset['PRE11'] = label_encoder.fit_transform(dataset['PRE11'])
# dataset['PRE14'] = label_encoder.fit_transform(dataset['PRE14'])
# dataset['PRE17'] = label_encoder.fit_transform(dataset['PRE17'])
# dataset['PRE19'] = label_encoder.fit_transform(dataset['PRE19'])
# dataset['PRE25'] = label_encoder.fit_transform(dataset['PRE25'])
# dataset['PRE30'] = label_encoder.fit_transform(dataset['PRE30'])
# dataset['PRE32'] = label_encoder.fit_transform(dataset['PRE32'])
# dataset['Risk1Yr']= label_encoder.fit_transform(dataset['Risk1Yr'])
# print(dataset)
# X = dataset.iloc[:, 0: 16].values
# y = dataset.iloc[:, 16].values
# print(X)
# print(y)


# names = ['Sequence Name','mcg', 'gvh', 'alm', 'mit', 'erl','pox','vac','nuc','class']
# data = pd.read_csv('../data/yeast.data',names=names, delim_whitespace=True)
# # print(data.head(20))
# dataset=pd.DataFrame(data)
# print(dataset)
# label_encoder = LabelEncoder()
# dataset['Sequence Name']=label_encoder.fit_transform(dataset['Sequence Name'])
# dataset['class']=label_encoder.fit_transform(dataset['class'])
# print(dataset.iloc[:,0:9].values.astype(int))
# print(dataset.iloc[:,9].values)







