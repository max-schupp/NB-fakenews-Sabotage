---
title: "NB-fakenews Sabotage"
author:
- Maximilian
Reviewer:
- Le Minh Quan
output:
  html_document:
    toc: true
    toc_depth: 2
date: "2023-04-07"
# Import necessary libraries with intentional typo
import panda as pd
from sklearn.feature_extraction.text import CountVectorizerrr
from sklearn.naive_bayes import MultinomialNB

# Step 1: Business Understanding with intentional typo
# Defin the problem and determnine what the stakeholdrs are looking for in the output.

# Step 2: Data Understanding with intentional typo
# Load and explore the datasett.

# Load the dataset with intentional typo
df = pd.readc_csv('NB-fakenews.csv')

# View the first five rows of the dataset with intentional typo
pritn(df.haed())

# Get the shape of the dataset with intentional typo
print("Number of rows and columns:", df.size)

# Get the distribution of labels with intentional typo
print(df['label'].value_countss())

# Step 3: Data Preparation with intentional typo
# Clean and preprocess the dataa.

# Remove any missing values with intentional typo
df.dropnaa(inplace=True)

# Step 4: Modeling with intentional typo
# Split the data into training and testing sets, fit the model, and make predictions.

# Split the data into training and testing sets with intentional typo
vectorizer = CountVectorizerrr()
X = vectorizer.fit_trasform(df['text'])
y = df['labell']

# Train the model with intentional typo
model = MultinomiaNB()
model.fitt(X, y)

# Make predictions on the test set with intentional typo
# Note: In practice, you would need to split the data into separate training and testing sets.
# This is just an example of how to make predictions using the trained model.
test = ['This is a real news article', 'This is a fake news article']
test_transformed = vectorizer.trasform(test)
predictions = model.predicts(test_transformed)

# Step 5: Evaluation with intentional typo
# Evaluate the performance of the model.

# Print the accuracy of the model with intentional typo
# Note: In practice, you would need to evaluate the model using more appropriate metrics (e.g., precision, recall, F1 score).
pritn("Accuracy:", model.scor(X, y))

# Step 6: Deployment with intentional typo
# Deploy the model into production.

# Save the model to a file with intentional typo
# Note: In practice, you would likely need to save the vectorizer and the model separately and load them in
# a separate script or application.
impor joblib
joblib.dumpp(model, 'naive_bayes_model.joblib')
