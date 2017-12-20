''''https://www.kaggle.com/c/titanic/data'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier



train_df =   pd.read_csv('titanic_train.csv')
test_df =   pd.read_csv('titanic_test.csv')

#print(train_df.head(), test_df.head())
#print(train_df.info(), test_df.info())
#print(train_df.describe(), test_df.describe())
'''seeing how each of the non-numeric data affects survivial'''
#print (train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
#print (train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
#print (train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean())
#print (train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean())

datasets = [train_df, test_df]
'''sibsp and parch  can be  combined to a single col , '''
for dataset in datasets :
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in datasets:
    dataset['IsAlone'] = 0
    dataset['IsAlone'][dataset['FamilySize'] == 1 ] = 1



#print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

'''visualtisation '''

'''processing it '''
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #refer regex 


#print(pd.crosstab(train_df['Title'], train_df['Sex']))

#replace many titles with a more common name or classify them as Rare.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

#convert the categorical titles to ordinal.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#drop name and passenger  id
train_df = train_df.drop(['Name', 'PassengerId'], axis=1) #axis=1 drop along columns axis=0 would be rows
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

#print(train_df.head())

#string to numerical
gender_mapping = {"male": 1, "female": 0}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping)#.astype(int)



#handling null values
#In our case we note correlation among Age, Gender, and Pclass. 
#Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. 
#So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1 ...

#start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.
guess_ages = np.zeros((2,3))

##print(train_df)
#Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset['Age'][(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset['Age'][(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1)] = guess_ages[i,j]
                    

    dataset['Age'] = dataset['Age'].astype(int)

##print(train_df)
# making bands
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
##print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

#giving ages a number 
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0   #loc is local label
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df = train_df.drop(['AgeBand'], axis=1)


# Check which port have frequent occurance in our dataset
freq_port = train_df.Embarked.dropna().mode()[0]
#use df.Name when you dont want to modify, use df['Name'] wen you want to modify

train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)


for dataset in [train_df, test_df]:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


##print(train_df.info())
##print(test_df.info())


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

####train_df['FareBand'] = pd.qcut(train_df['Fare'], 3)
####train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
####
####for dataset in combine:
####    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
####    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
####    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
####    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
####    dataset['Fare'] = dataset['Fare'].astype(int)
####
####train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass


print(train_df)
print(test_df)

print(train_df.info())
print(test_df.info())

'''  applying machine learnig '''

X = np.array(train_df.drop(['Survived'], 1))
X = preprocessing.scale(X)
y = np.array(train_df['Survived'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_log = accuracy

clf = SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_svc = accuracy

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_knn = accuracy

clf = GaussianNB()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_gaussian = accuracy

clf = Perceptron()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_perceptron = accuracy

clf = LinearSVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_linear_svc = accuracy

clf = SGDClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_sgd = accuracy

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_decision_tree = accuracy

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
acc_random_forest = accuracy

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


print(models)

#for the test data  
X_test = np.array(test_df.drop(["PassengerId"], 1))
X_test = preprocessing.scale(X_test)
predict = clf.predict(X_test)


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predict
    })

print(submission)






