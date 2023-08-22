import pandas as pd

# Read the text file into a DataFrame
df = pd.read_csv('validation.txt', sep='&', skipinitialspace=True, engine='python')

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Function to clean and convert values to float
def clean_and_convert(value):
    try:
        if type(value) is not float :
            # Remove backslashes and leading/trailing spaces, then convert to float
            value = float(value.replace('\\', '').strip())
        return value
    except ValueError:
        # If conversion fails, return NaN
        return float('nan')

# Apply the clean_and_convert function to each column containing metric values
for col in df.columns[1:]:
    df[col] = df[col].apply(clean_and_convert)

# Define a function to compute custom fitness
def custom_fitness(row):
    # Define the weights for each metric
    weights = [0.25, 0.23, 0.15, 0.15, 0.22]

    # Extract the metric values from the row and convert them to float
    metrics = row[['mAP@50-95', 'mAP@50', 'mAP@75', 'mP', 'mR']].astype(float).values

    # Compute the custom fitness score as the weighted sum of metrics
    fitness = sum(weights[i] * metrics[i] for i in range(len(metrics)))

    return fitness

# Apply the custom_fitness function to each row in the DataFrame
df['Fitness'] = df.apply(custom_fitness, axis=1)

# Sort the DataFrame by the custom fitness score in descending order
sorted_df = df.sort_values(by='Fitness', ascending=False)

# Print the best 4 indices with their corresponding fitness scores
best_indices = sorted_df[['index', 'Fitness']].head(10)
print("Best 4 Indices with Corresponding Fitness Scores:")
print(best_indices.to_string(index=False))
