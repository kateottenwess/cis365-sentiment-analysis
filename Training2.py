import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset and sample a fraction
df = pd.read_csv("C:/Users/kcree/OneDrive/Desktop/AI/Training.csv")
df_sampled = df.sample(frac=0.1, random_state=42)

# Fill NaN values in the 'comment_text' column with empty strings
df_sampled['comment_text'].fillna('', inplace=True)

# Remove rows with NaN in any of the target label columns
df_sampled.dropna(subset=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'not_toxic'], inplace=True)

X = df_sampled['comment_text']  # Text comments
y = df_sampled[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'not_toxic']]  # Labels

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training a multi-label classifier using Logistic Regression
model = MultiOutputClassifier(LogisticRegression(random_state=42))
model.fit(X_train_tfidf, y_train)

# Making predictions
predictions = model.predict_proba(X_test_tfidf)

# Converting predictions to a DataFrame (for better readability)
probabilities = pd.DataFrame({label: preds[:, 1] for label, preds in zip(y.columns, predictions)})

# Add the original comments to the probabilities DataFrame
results = X_test.reset_index(drop=True).to_frame().join(probabilities)

# Write the results to a CSV file in a different directory to avoid permission issues
results.to_csv("C:/Users/kcree/OneDrive/Desktop/AI/predictionsLogic.csv", index=False)

# Print the comment along with the predicted probabilities
print(results.head())

##################################################################
#Save vectorizer requirments
joblib.dump(model, 'toxic_comment_model.joblib')
joblib.dump(vectorizer, 'toxic_comment_vectorizer.joblib')

vectorizer_path = 'C:/Users/kcree/Documents/ToxicModel/toxic_comment_vectorizer.joblib'
model_path = 'C:/Users/kcree/Documents/ToxicModel/toxic_comment_model.joblib'


# Load preprocessed data
new_data_path = "C:/Users/kcree/Documents/ToxicModel/cleaned_data.csv"
new_data_df = pd.read_csv(new_data_path)

# Print column names for debugging
print("Columns available in the DataFrame:", new_data_df.columns)

# Check if 'cleaned_text' column exists
if 'cleaned_text' not in new_data_df.columns:
    print("Column 'cleaned_text' not found. Please check the CSV file for the correct column name.")
    exit()  # Exit the script if the column isn't found

# Proceed with replacing NaN values and converting to string
new_data_df['cleaned_text'].fillna('', inplace=True)
new_comments = new_data_df['cleaned_text'].astype(str)

# Transform the new data using the fitted vectorizer
new_comments_transformed = vectorizer.transform(new_comments)


# Make predictions on the new data
new_predictions = model.predict_proba(new_comments_transformed)

# Converting new predictions to a DataFrame for better readability
new_probabilities = pd.DataFrame({label: preds[:, 1] for label, preds in zip(y.columns, new_predictions)})

# Add the original comments to the new probabilities DataFrame
new_results = new_data_df.reset_index(drop=True).join(new_probabilities)

# Write the new results to a CSV file
output_path = "C:/Users/kcree/Documents/ToxicModel/FinalPredictions.csv"  # Update this path as needed
new_results.to_csv(output_path, index=False)

# Inform that the process is complete
print(f"Predictions written to {output_path}")

