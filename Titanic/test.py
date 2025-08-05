from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def get_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Calculate accuracy for RandomForestClassifier with given max_leaf_nodes"""
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    accuracy = accuracy_score(val_y, preds_val)
    return accuracy

def find_best_leaf_nodes(train_X, val_X, train_y, val_y, leaf_node_options=[5, 50, 500, 5000]):
    """Test different max_leaf_nodes values and return the best one based on accuracy"""
    results = {}
    best_accuracy = 0
    best_nodes = None
    
    print("Testing different max_leaf_nodes values:")
    print("-" * 50)
    
    for max_leaf_nodes in leaf_node_options:
        accuracy = get_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y)
        results[max_leaf_nodes] = accuracy
        
        print(f"Max leaf nodes: {max_leaf_nodes:4d} \t Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_nodes = max_leaf_nodes
    
    print("-" * 50)
    print(f"Best max_leaf_nodes: {best_nodes} with Accuracy: {best_accuracy:.4f}")
    
    return best_nodes, results

# Data Loading and Processing
try:
    # Load data
    train_clean = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_train.csv')
    test_clean = pd.read_csv('C:/Users/DEKATECH/Kaggle submissions/Titanic/cleaned_test.csv')
    
    print(f"Loaded training data shape: {train_clean.shape}")
    print(f"Loaded test data shape: {test_clean.shape}")
    
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please check if the file paths are correct.")
    exit()

# Drop PassengerId from training data
if 'PassengerId' in train_clean.columns:
    train_clean.drop('PassengerId', axis=1, inplace=True)

# Check for required columns in training data
required_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
missing_columns = [col for col in required_columns if col not in train_clean.columns]

if missing_columns:
    print(f"Missing required columns in training data: {missing_columns}")
    exit()

# Filter rows with missing values from training data
print(f"Original training data shape: {train_clean.shape}")
filtered_train_data = train_clean.dropna(axis=0)
print(f"After dropping missing values: {filtered_train_data.shape}")

if len(filtered_train_data) == 0:
    print("No data remaining after dropping missing values!")
    exit()

# Prepare training data
y = filtered_train_data['Survived']
train_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = filtered_train_data[train_features].copy()

# Handle categorical variables
cat_cols = ['Sex', 'Embarked']
ordinal_encoder = OrdinalEncoder()
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols])

print(f"Final training data shape: {X.shape}")
print("Categorical columns encoded:", cat_cols)

# Split the training data into train/validation for hyperparameter tuning
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"Training subset size: {len(train_X)}")
print(f"Validation subset size: {len(val_X)}")

# Test different max_leaf_nodes values
print("\n" + "="*60)
print("TESTING DIFFERENT MAX_LEAF_NODES VALUES")
print("="*60)

# Find the best number of leaf nodes
best_nodes, all_results = find_best_leaf_nodes(train_X, val_X, train_y, val_y)

print(f"\n" + "="*60)
print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
print("="*60)

# Train final model on ALL training data with the best max_leaf_nodes
final_model = RandomForestClassifier(max_leaf_nodes=best_nodes, n_estimators=100, random_state=0)
final_model.fit(X, y)  # Use ALL training data

print(f"Final model trained with max_leaf_nodes={best_nodes} and n_estimators=100")
print(f"Model trained on {len(X)} samples")

# Optional: Prepare test data for predictions
print("\nPreparing test data for predictions...")

# Clean test data similarly
test_features = test_clean[train_features].copy()

# Apply same encoding to test data
if all(col in test_clean.columns for col in cat_cols):
    test_features[cat_cols] = ordinal_encoder.transform(test_features[cat_cols])
    print("Test data encoded successfully")
    
    # Handle missing values in test data (drop rows with missing values)
    print(f"Test data shape before handling missing values: {test_features.shape}")
    test_features_clean = test_features.dropna()
    print(f"Test data shape after dropping missing values: {test_features_clean.shape}")
    
    # Make predictions on test data
    test_predictions = final_model.predict(test_features_clean)
    print(f"Generated {len(test_predictions)} predictions for test data")
    
    # Create submission file
    if 'PassengerId' in test_clean.columns:
        test_clean_for_submission = test_clean.dropna(subset=train_features)
        submission = pd.DataFrame({
            'PassengerId': test_clean_for_submission['PassengerId'],
            'Survived': test_predictions
        })
        print("Submission dataframe created")
        # submission.to_csv('titanic_submission.csv', index=False)
else:
    print("Warning: Test data missing required categorical columns")

print("\nProcess completed!")