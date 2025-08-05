#Import libraries
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


#Load the dataset
train_df = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/train.csv')
test_df = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/test.csv')

#Perform data cleaning
#For train.csv, drop unnecessary columns
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
#Check columns with missing values
cols_with_msg_val = [col for col in train_df.columns if train_df[col].isnull().any()]
#print(f"Columns with missing values: {cols_with_msg_val}")      #'Age', 'Embarked'
#Drop rows with missing data in the specified columns
train_df.dropna(subset=cols_with_msg_val, inplace=True)
#For test.csv, drop unnecessary columns
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
#Check columns with missing values in test set
cols_with_msg_val1 = [col for col in test_df.columns if test_df[col].isnull().any()]
#print(f"Columns with missing values in test set: {cols_with_msg_val1}")     #'Age', 'Fare'
#Fill missing values with median
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
# train_df.to_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_train.csv', index=False)
# test_df.to_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_test.csv', index=False)

#Import the cleaned data
train_clean = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_train.csv')
test_clean = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_test.csv')
train_clean.drop('PassengerId', axis=1, inplace=True)
test_clean.drop('PassengerId', axis=1, inplace=True)
#Encode categorical variables
ordinal_encoder = OrdinalEncoder()
cat_cols = ['Sex', 'Embarked']
train_clean[cat_cols] = ordinal_encoder.fit_transform(train_clean[cat_cols])
test_clean[cat_cols] = ordinal_encoder.transform(test_clean[cat_cols])

# train_clean.to_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_train1.csv', index=False)

#Extract features and target
X_train = train_clean.drop('Survived', axis=1)
y_train = train_clean['Survived']
X_test = test_clean.copy()
y_test = test_clean['Survived'] if 'Survived' in test_clean.columns else None

#Plot each feature against the target
#Pclass vs Survived
plt.figure(figsize=(10, 6))
pclass_survival = train_clean.groupby('Pclass')['Survived'].mean()
plt.bar([1, 2, 3], pclass_survival.values, color=['gold', 'silver', 'brown'], alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks([1, 2, 3], ['1st Class', '2nd Class', '3rd Class'])
plt.grid(axis='y', alpha=0.3)
plt.text(1, pclass_survival[1] + 0.02, f'{pclass_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, pclass_survival[2] + 0.02, f'{pclass_survival[2]:.3f}', ha='center', fontweight='bold')
plt.text(3, pclass_survival[3] + 0.02, f'{pclass_survival[3]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#Sex vs Survived
plt.figure(figsize=(10, 6))
sex_survival = train_clean.groupby('Sex')['Survived'].mean()
plt.bar(['female', 'male'], sex_survival.values, color=['lightcoral', 'lightblue'], alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.text(0, sex_survival[0] + 0.02, f'{sex_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, sex_survival[1] + 0.02, f'{sex_survival[1]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#Age vs Survived
plt.figure(figsize=(12, 6))
age_bins = [0, 12, 18, 30, 50, 80]
age_labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-30)', 'Adult (31-50)', 'Senior (50+)']
train_clean['Age_Group'] = pd.cut(train_clean['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
age_survival = train_clean.groupby('Age_Group')['Survived'].mean()
plt.bar(age_labels, age_survival.values, color='lightgreen', alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.text(0, age_survival[0] + 0.02, f'{age_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, age_survival[1] + 0.02, f'{age_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, age_survival[2] + 0.02, f'{age_survival[2]:.3f}', ha='center', fontweight='bold')
plt.text(3, age_survival[3] + 0.02, f'{age_survival[3]:.3f}', ha='center', fontweight='bold')
plt.text(4, age_survival[4] + 0.02, f'{age_survival[4]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#SibSp vs Survived
plt.figure(figsize=(10, 6))
sibsp_survival = train_clean.groupby('SibSp')['Survived'].mean()
sibsp_counts = sibsp_survival.index.tolist()
plt.bar(sibsp_counts, sibsp_survival.values, color='orange', alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Number of Siblings/Spouses', fontsize=16, fontweight='bold')
plt.xlabel('Number of Siblings/Spouses', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.text(0, sibsp_survival[0] + 0.02, f'{sibsp_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, sibsp_survival[1] + 0.02, f'{sibsp_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, sibsp_survival[2] + 0.02, f'{sibsp_survival[2]:.3f}', ha='center', fontweight='bold')
if 3 in sibsp_survival.index:
    plt.text(3, sibsp_survival[3] + 0.02, f'{sibsp_survival[3]:.3f}', ha='center', fontweight='bold')
if 4 in sibsp_survival.index:
    plt.text(4, sibsp_survival[4] + 0.02, f'{sibsp_survival[4]:.3f}', ha='center', fontweight='bold')
if 5 in sibsp_survival.index:
    plt.text(5, sibsp_survival[5] + 0.02, f'{sibsp_survival[5]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#Parch vs Survived
plt.figure(figsize=(10, 6))
parch_survival = train_clean.groupby('Parch')['Survived'].mean()
parch_counts = parch_survival.index.tolist()
plt.bar(parch_counts, parch_survival.values, color='purple', alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Number of Parents/Children', fontsize=16, fontweight='bold')
plt.xlabel('Number of Parents/Children', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.text(0, parch_survival[0] + 0.02, f'{parch_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, parch_survival[1] + 0.02, f'{parch_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, parch_survival[2] + 0.02, f'{parch_survival[2]:.3f}', ha='center', fontweight='bold')
if 3 in parch_survival.index:
    plt.text(3, parch_survival[3] + 0.02, f'{parch_survival[3]:.3f}', ha='center', fontweight='bold')
if 4 in parch_survival.index:
    plt.text(4, parch_survival[4] + 0.02, f'{parch_survival[4]:.3f}', ha='center', fontweight='bold')
if 5 in parch_survival.index:
    plt.text(5, parch_survival[5] + 0.02, f'{parch_survival[5]:.3f}', ha='center', fontweight='bold')
if 6 in parch_survival.index:
    plt.text(6, parch_survival[6] + 0.02, f'{parch_survival[6]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#Fare vs Survived
plt.figure(figsize=(12, 6))
fare_bins = [0, 10, 30, 50, 100, 600]
fare_labels = ['Low (0-10)', 'Medium (10-30)', 'High (30-50)', 'Premium (50-100)', 'Luxury (100+)']
train_clean['Fare_Group'] = pd.cut(train_clean['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
fare_survival = train_clean.groupby('Fare_Group')['Survived'].mean()
plt.bar(fare_labels, fare_survival.values, color='teal', alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Fare Range', fontsize=16, fontweight='bold')
plt.xlabel('Fare Range', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.text(0, fare_survival[0] + 0.02, f'{fare_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, fare_survival[1] + 0.02, f'{fare_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, fare_survival[2] + 0.02, f'{fare_survival[2]:.3f}', ha='center', fontweight='bold')
plt.text(3, fare_survival[3] + 0.02, f'{fare_survival[3]:.3f}', ha='center', fontweight='bold')
plt.text(4, fare_survival[4] + 0.02, f'{fare_survival[4]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()
#Embarked vs Survived
plt.figure(figsize=(10, 6))
embarked_survival = train_clean.groupby('Embarked')['Survived'].mean()
port_labels = ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)']
plt.bar(port_labels, embarked_survival.values, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
plt.title('Survival Rate by Port of Embarkation', fontsize=16, fontweight='bold')
plt.xlabel('Port of Embarkation', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.text(0, embarked_survival[0] + 0.02, f'{embarked_survival[0]:.3f}', ha='center', fontweight='bold')
plt.text(1, embarked_survival[1] + 0.02, f'{embarked_survival[1]:.3f}', ha='center', fontweight='bold')
plt.text(2, embarked_survival[2] + 0.02, f'{embarked_survival[2]:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
#plt.show()

#Build and fit the model
#Model1 - RandomForestClassifier
model = RandomForestClassifier(max_leaf_nodes=50, n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
#print(f"Predictions: {preds}")

#Check for accuracy
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")

#Extract submission data
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds})
submission.to_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/submission.csv', index=False)