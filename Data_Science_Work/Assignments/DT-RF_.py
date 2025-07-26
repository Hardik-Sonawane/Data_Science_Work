#########################################################
#####    Decision Tree nd Random Forest #####################
########################################################

"""Que.1
1.Business Problem
1.1.What is the business objective?
1.1.Are there any constraints?
"""
#Solution:-
#The goal is to use Decision Trees or Random Forests to predict things like:
#Customer Purchases: Helping stores know what customers might buy to increase sales.
#Customer Groups: Dividing customers into groups to send them the right advertisements.
#Risk Assessment: Helping banks figure out how likely someone is to repay a loan.
#1.2. Are there any constraints?
#Some challenges might be:

#Data Quality: We need good, clean data for accurate predictions.
#Computational Resources: Random Forests need more computer power than Decision Trees.
#Interpretability: Decision Trees are easier to understand, while Random Forests can be complex and harder to explain.
#Time and Budget: Limited time and money may affect how much data we can use and how detailed our models can be

##################################################
"""
QUE.2.Work on each feature of the dataset to create a data dictionary as displayed in the below image:
 2.1 Make a table as shown above and provide information about the features such as its data type and its relevance to the model building. And if not relevant, provide reasons and a description of the feature.  
"""
#Solution 
# i've solve this by taking a smaple data
"""
Customer ID  Age  Gender  Annual Income  Purchase Amount Loyalty Status
0            1   25    Male          60000             1500            Yes
1            2   34  Female          75000             2000             No
2            3   29  Female          50000             1000            Yes
3            4   42    Male         120000             5000             No
4            5   30  Female          80000             3000            Yes
"""
# and here's the explanation 
#Customer ID is marked as irrelevant because it is merely an identifier and does not provide useful information 
#for predictive modeling.
#Age, Gender, Annual Income, Purchase Amount, and Loyalty Status are all relevant features as
# they provide insights that can help predict customer purchasing behavior
##################################################################################

"""
Que.3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.
"""
#Solution:-

#3. Data Pre-processing for Decision Trees and Random Forests
#3.1 Data Cleaning
#Handle Missing Values: Fill in or remove missing data to ensure complete datasets.
#Remove Duplicates: Eliminate duplicate entries to prevent skewing the results.
#Detect Outliers: Identify and manage outliers to improve model accuracy, even though Decision Trees and Random Forests are somewhat robust to them.
#Ensure Data Type Consistency: Check that each feature is in the correct format (e.g., numerical or categorical).

#3.2 Feature Engineering
#Create New Features: Develop additional features (like spending categories) to capture more insights from the data.
#Encode Categorical Variables: Convert categories into numerical values (using one-hot or label encoding) for better model performance.
#Consider Feature Scaling: While not necessary for Decision Trees, scaling can be useful if combining with other models.
#Select Relevant Features: Identify and retain only the features that contribute significantly to model performance.

####################################################################################

"""
QUE.4.	Exploratory Data Analysis (EDA):
4.1.	Summary.
4.2.	Univariate analysis.
4.3.	Bivariate analysis.
"""
#Solution:-


#4. Exploratory Data Analysis (EDA)
#4.1 Summary
#In this step, we get a quick overview of the dataset:

#Descriptive Statistics: Look at basic numbers like average (mean), middle value (median), most common value (mode), and how spread out the numbers are (standard deviation).
#Data Types: Identify what type each feature is (like numbers or categories).
#Missing Values: Check for any missing data and how much of it there is.

#4.2 Univariate Analysis
#Here, we focus on one feature at a time to see how it looks:

#Histograms: These are bar graphs that show how often different values occur in a numerical feature. They help us see the shape of the data (like if it's balanced or skewed).
#Box Plots: These plots show the spread of numerical data and highlight any outliers (values that are way higher or lower than most).
#Frequency Counts: For categorical features, we count how many times each category appears to understand their distribution.

#4.3 Bivariate Analysis
#In this part, we look at two features together to see how they relate:

#Scatter Plots: We create plots that show one numerical feature against another to see if there's a pattern or relationship between them (like if older customers spend more).
#Correlation Matrix: This is a table that shows how strongly different numerical features are related. A high number means they are closely linked.
#Group Comparisons: We can compare how a numerical feature behaves across different categories using box plots or bar charts (for example, comparing spending amounts by customer type).
##############################################################################
"""
Que.5.	Model Building
5.1	Build the model on the scaled data (try multiple options).
5.2	Perform Decision Tree and Random Forest on the given datasets.
5.3	Train and Test the data and perform cross validation techniques, compare accuracies, precision and recall and explain about them.
5.4	Briefly explain the model output in the documentation. 
"""
#Solution:-


#5. Model Building
#5.1 Build the Model on Scaled Data
#Scaling the Data: Before diving into model building, I’ll start by scaling the features in my dataset. Scaling ensures that no single feature 
#dominates the others due to differences in their ranges. This is a crucial step that helps improve model performance.
#Trying Different Models: I’ll experiment with different models to see which one works best. I plan to use:
#Decision Tree: This model will help me understand how decisions are made based on the features. I find decision trees intuitive because they visually represent the decision-making process.
#Random Forest: I’ll also implement a random forest model, which uses multiple decision trees to enhance accuracy and reduce overfitting. It’s like getting a consensus from several trees to make predictions!

#5.2 Perform Decision Tree and Random Forest
#Decision Tree: I’ll begin by building a decision tree model using the training data. This will show me how different features influence the outcomes. I’ll visualize the tree to interpret it easily.
#Random Forest: After that, I’ll set up the random forest model. This approach should give me a more reliable prediction since it averages the results
# from several decision trees. I’m expecting it to outperform the single decision tree model.

#5.3 Train and Test the Data
#Split the Data: I’ll split my dataset into training and testing sets, using about 70-80% of the data for training. This allows me to train the models and then see how well they generalize on unseen data.
#Cross-Validation: To validate my models, I’ll use k-fold cross-validation. This will help me ensure that my model is robust and not just memorizing the training data. It’s important
# to assess how well my model performs across different subsets of the data.
#Compare Metrics: After training, I’ll look at different performance metrics to compare the models:
#Accuracy: I’ll calculate the overall accuracy to see how many predictions were correct.
#Precision: I’ll also check precision to understand how many of the predicted positive cases were actually positive.
#Recall: Lastly, I’ll look at recall to evaluate how many actual positive cases I correctly identified. Analyzing these metrics will give 
#me a better idea of which model performs best and how they each contribute to my analysis.

#5.4 Explain the Model Output
#Model Results: Finally, I’ll summarize the results from my models. I’ll present the accuracy, precision, and recall for both the decision tree and random forest models. I’ll highlight which features had the most 
#significant impact on predictions and discuss
# how these insights could be applied in real-world scenarios. If there are areas for improvement, I’ll mention those too, such as tuning hyperparameters or trying other algorithms.

###########################################################
"""
QUE.6
6.Write about the benefits/
impact of the solution - in what way does the business (client)
 benefit from the solution provided?
"""
# Solution:-

#6. Benefits/Impact of the Solution
#In this section, I’ll discuss how the Decision Tree and Random Forest models can benefit the business:

#Informed Decision-Making: By leveraging these models, the business can make better decisions based on actual data insights. This is crucial for effective strategic planning.

#Understanding Customers: The models help uncover what drives customer behavior. With this knowledge, the business can tailor its offerings and marketing efforts to better serve its customers.

#Increased Efficiency: Automating predictions reduces manual analysis time, allowing the team to allocate resources to other important areas, like improving products or enhancing customer support.

#Risk Mitigation: The insights from these models can highlight potential risks before they become significant issues, helping the business to take preventive actions.

#Gaining Competitive Advantage: Using advanced analytics like Decision Trees and Random Forests positions the business ahead of competitors who may not be utilizing similar data-driven methods.

#Scalability: The models are easily adaptable and can be updated with new data, which is vital for keeping up with changing market conditions.
###################################################################################


###########################################################################
#################### Problem Statements #########################################
#####################################################################
"""
PS--1.	A cloth manufacturing company is interested to know 
about the different attributes contributing to high sales. Build a decision tree & random forest
 model with Sales as target variable (first convert it into categorical variable).

"""
#Solution:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r"C:\13-decision_tree\Company_Data.csv"
df = pd.read_csv(file_path)

# Check the data structure and for NaN values
print("Dataset Overview:")
print(df.head())
print("\nData Information:")
print(df.info())

# Check for NaN values
print("\nChecking for NaN values:")
print(df.isnull().sum())

# Handling NaN values - Drop rows with NaN values
df.dropna(inplace=True)

# Convert 'Sales' to a categorical variable
bins = [0, 5, 10, 15]
labels = ['Low', 'Medium', 'High']
df['Sales_Category'] = pd.cut(df['Sales'], bins=bins, labels=labels, right=False)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

# Split the data into features and target variable
X = df_encoded.drop(['Sales', 'Sales_Category'], axis=1)
y = df_encoded['Sales_Category']

# Ensure all data is numeric
print("\nData Types Before Conversion:")
print(X.dtypes)

# Convert any non-numeric columns to numeric if necessary
# (This is usually not needed if you've done one-hot encoding correctly)
# Uncomment the next line if there are any object types remaining
# X = X.apply(pd.to_numeric, errors='coerce')

# Final check for NaN values
print("\nFinal NaN Check:")
print(X.isnull().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Fit the model
try:
    dt_model.fit(X_train, y_train)
except Exception as e:
    print("Error fitting Decision Tree model:", e)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model
try:
    rf_model.fit(X_train, y_train)
except Exception as e:
    print("Error fitting Random Forest model:", e)

# Predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate models
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_predictions))

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions))


##################################################################################
##############################################################################

"""
PS--2.	 Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome
 as the output variable.

"""
# Solution:-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = r"C:/13-decision_tree/Diabetes.csv"  # Update to your actual file path
df = pd.read_csv(file_path)

# Print the column names to check their correctness
print("Column names:", df.columns)

# Assuming you see the column name for the outcome variable,
# Replace 'Class variable' with the actual name you see in the output
# Map 'YES' and 'NO' in the appropriate column to binary values in a new column 'Outcome'
# Adjust the name based on what you find in the printed output
outcome_column_name = 'Class variable'  # Change this if needed based on the printed output
df['Outcome'] = df[outcome_column_name].apply(lambda x: 1 if x == 'YES' else 0)

# Drop the original outcome column
df = df.drop(columns=[outcome_column_name])

# Split the data into features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Build and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate the models
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_predictions))

print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions))

#############################################################################
#############################################################################

"""
PS--3.	Build a Decision Tree & Random 
Forest model on the fraud data. Treat 
those who have taxable_income <= 30000
 as Risky and others as Good (discretize the taxable 
                              """
#Solution:-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
file_path = "C:/13-decision_tree/Fraud_check.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())

# Step 2: Discretize the `Taxable.Income` column to classify into Risky/Good
df['Risk'] = df['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

# Check the first few rows after adding the 'Risk' column
print("\nDataset with Risk classification:")
print(df[['Taxable.Income', 'Risk']].head())

# Step 3: Encode categorical variables (Undergrad, Urban, and Marital.Status)
df['Undergrad'] = df['Undergrad'].apply(lambda x: 1 if x == 'YES' else 0)
df['Urban'] = df['Urban'].apply(lambda x: 1 if x == 'YES' else 0)
df['Marital.Status'] = df['Marital.Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

# Display the first few rows to confirm encoding
print("\nDataset after encoding categorical variables:")
print(df.head())

# Step 4: Split the data into features (X) and target variable (y)
X = df[['Undergrad', 'Marital.Status', 'City.Population', 'Work.Experience', 'Urban']]
y = df['Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Build and evaluate the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
print("\nDecision Tree Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Step 6: Build and evaluate the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


#############################################################################################3

"""
PS- 4.	In the recruitment domain, HR faces the challenge of
 predicting if the candidate is faking their
 salary or not. For example, a candidate 
 claims to have 5 years of experience and earns 70,000 per month working 
 as a regional manager. The candidate expects more money than his
 previous CTC. We need a way to verify their claims (is 70,000 a month working as a regional manager with an experience 
         of 5 years a genuine claim or does he/she make 
less than that?) Build a Decision Tree and Random Forest model with monthly income as the target variable.
"""

#Solution:-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_path = "C:/13-decision_tree/HR_DT.csv"  # Correct file path
df = pd.read_csv(file_path)

# Step 2: Check the column names and display the dataset
print("Column Names in the Dataset:")
print(df.columns)

# Clean column names by removing any leading or trailing spaces
df.columns = df.columns.str.strip()

# Display the first few rows of the dataset to check its structure
print("\nDataset:")
print(df.head())

# Step 3: Preprocessing the dataset
# Convert categorical variable (Position) to numeric using Label Encoding
# Ensure the column name is correct here
if 'Position of the employee' in df.columns:
    label_encoder = LabelEncoder()
    df['Position'] = label_encoder.fit_transform(df['Position of the employee'])
else:
    print("Column 'Position of the employee' not found!")

# Display the transformed dataset
print("\nDataset after label encoding Position:")
print(df.head())

# Step 4: Check for the correct column name for experience
print("\nChecking the column names after cleanup:")
print(df.columns)

# Step 5: Split the data into features (X) and target variable (y)
# Make sure we are using the correct column names
if 'No of Years of Experience of employee' in df.columns and 'monthly income of employee' in df.columns:
    X = df[['Position', 'No of Years of Experience of employee']]  # Features
    y = df['monthly income of employee']  # Target variable
else:
    print("Required columns are missing or incorrectly named.")

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Build and evaluate the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
print("\nDecision Tree Model Evaluation:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_dt)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt)}")
print(f"R-squared: {r2_score(y_test, y_pred_dt)}")

# Step 8: Build and evaluate the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model Evaluation:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")


#########################################################################


### Conclusion
"""
In this assignment, we used **Decision Tree** and **Random Forest**
 models to predict salary claims based on position 
 and experience, aiming to help HR validate 
 candidates' salary expectations. After
 preprocessing the data, we found that 
 **Random Forest** outperformed **Decision Tree** 
 in accuracy. The model can automate salary
 verification, saving time and improving 
 decision-making. This project reinforced
 my understanding of the machine learning
 process and its real-world business applications.
 """
 
 ##################################################
 #############################################################################
 ##########################################################################################
 ###############################################################
 ####################################################################################################################