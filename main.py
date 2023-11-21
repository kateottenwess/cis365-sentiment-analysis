import pandas as pd
import string

# Load the data
file_path = "C:/Users/kcree/Documents/ToxicModel/FinalTesting.xlsx" # Change to your file path
data_df = pd.read_excel(file_path)

# Preprocessing function
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Handle non-string data
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = text.encode('ascii', 'ignore').decode()  # Remove non-ascii characters
    return text

# Apply the preprocessing
data_df['cleaned_text'] = data_df['comment_text'].apply(clean_text)

# Save the cleaned text to a new file
output_file = 'C:/Users/kcree/Documents/ToxicModel/cleaned_data.csv'  # The file name for the cleaned data
data_df[['cleaned_text']].to_csv(output_file, index=False)

print("Cleaning complete. Data saved to:", output_file)
