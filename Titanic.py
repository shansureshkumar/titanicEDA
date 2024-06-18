# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())

# Step 3: Data Exploration and Cleaning

# Display basic information about the dataset
print(titanic.info())

# Check for missing values
print(titanic.isnull().sum())

# Fill missing values for 'age' with the median value
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill missing values for 'embarked' with the mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop columns 'deck' and 'embark_town' as they have many missing values
titanic.drop(columns=['deck', 'embark_town'], inplace=True)

# Step 4: Exploratory Data Analysis (EDA)

# 4.1. Survival Rate
# Calculate survival rate
survival_rate = titanic['survived'].mean()
print(f'Survival Rate: {survival_rate:.2f}')

# Plot survival count overall
fig=plt.figure(figsize=(6,6))
titanic['survived'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title('Overall Survival Rate')
plt.show()

# 4.2. Survival Rate by Class
# Survival rate by class
sns.barplot(x='pclass', y='survived', data=titanic)
plt.title('Survival Rate by Class')
plt.show()

# 4.3. Survival Rate by Gender
# Survival rate by gender
sns.barplot(x='sex', y='survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.show()

# 4.4. Age Distribution
# Age distribution
sns.histplot(titanic['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 4.5. Survival Rate by Age
# Survival rate by age
sns.scatterplot(x='age', y='survived', data=titanic)
plt.title('Survival Rate by Age')
plt.show()

# Step 5: Correlation Matrix
# Convert categorical variables to numeric representations
titanic_encoded = titanic.copy()
titanic_encoded['sex'] = titanic_encoded['sex'].map({'male': 0, 'female': 1})
titanic_encoded['embarked'] = titanic_encoded['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
titanic_encoded['class'] = titanic_encoded['class'].map({'First': 1, 'Second': 2, 'Third': 3})
titanic_encoded['who'] = titanic_encoded['who'].map({'man': 0, 'woman': 1, 'child': 2})
titanic_encoded['adult_male'] = titanic_encoded['adult_male'].astype(int)
titanic_encoded['alone'] = titanic_encoded['alone'].astype(int)
titanic_encoded['alive'] = titanic_encoded['alive'].map({'no': 0, 'yes': 1})

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(titanic_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 6: Pair Plot
# Pair plot
sns.pairplot(titanic_encoded, hue='survived')
plt.show()

# Step 7: Survival Rate by Family Size
# Create a new feature 'family_size'
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Survival rate by family size
sns.barplot(x='family_size', y='survived', data=titanic)
plt.title('Survival Rate by Family Size')
plt.show()
