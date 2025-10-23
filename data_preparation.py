import pandas as pd
import re

# Load the dataset with a fallback encoding
try:
    df = pd.read_csv('data/complete_laptop_data0.csv', encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 encoding failed, trying latin1...")
    df = pd.read_csv('data/complete_laptop_data0.csv', encoding='latin1')

# Select key columns for RAG system
key_columns = [
    'name', 'Price', 'Processor Brand', 'Processor Name', 'RAM', 'SSD Capacity',
    'Graphic Processor', 'Screen Size', 'Screen Resolution', 'Additional Features', 'user rating'
]
df = df[key_columns]

# Clean and normalize data
# 1. Handle missing values
# Drop rows where critical fields are missing
critical_columns = ['name', 'Processor Name', 'Price']
df = df.dropna(subset=critical_columns)

# Impute missing 'user rating' with median or placeholder (e.g., 3.0)
df['user rating'] = df['user rating'].fillna(df['user rating'].median() if df['user rating'].notna().sum() > 0 else 3.0)

# Impute missing 'Additional Features' with a default description
df['Additional Features'] = df['Additional Features'].fillna('No additional features provided.')

# 2. Normalize formats
# Clean Price: Remove currency symbols, commas, and convert to numeric
def clean_price(price):
    if isinstance(price, str):
        price = re.sub(r'[^\d]', '', price)  # Remove non-digits
        return pd.to_numeric(price, errors='coerce') / 100  # Convert to INR
    return price

df['Price'] = df['Price'].apply(clean_price)

# Standardize RAM and SSD Capacity to GB
def standardize_memory(value):
    if pd.isna(value):
        return value
    value = str(value).lower().strip()
    if 'tb' in value:
        return float(re.sub(r'[^\d.]', '', value)) * 1024  # Convert TB to GB
    return float(re.sub(r'[^\d.]', '', value))  # Keep GB or numeric

df['RAM'] = df['RAM'].apply(standardize_memory)
df['SSD Capacity'] = df['SSD Capacity'].apply(standardize_memory)

# Clean text fields (remove extra spaces, normalize case)
df['name'] = df['name'].str.strip().str.title()
df['Processor Brand'] = df['Processor Brand'].str.strip().str.title()
df['Processor Name'] = df['Processor Name'].str.strip().str.title()
df['Graphic Processor'] = df['Graphic Processor'].str.strip().str.title()
df['Additional Features'] = df['Additional Features'].str.strip()

# 3. Simulate reviews (placeholder for now)
# Use 'Additional Features' as description; simulate a review based on it
df['simulated_reviews'] = df['Additional Features'].apply(
    lambda x: f"This laptop has the following features: {x.lower()}. Users find it reliable and suitable for its intended purpose."
)

# 4. Save cleaned dataset
df.to_csv('data/cleaned_laptop_data.csv', index=False)

# Print summary
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isna().sum()}")
print(f"Sample rows:\n{df.head()}")