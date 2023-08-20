import pandas as pd
import matplotlib.pyplot as plt

# Read the text file into a DataFrame
df = pd.read_csv('validation.txt', sep='&', skipinitialspace=True, engine='python')

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Define a colormap
colormap = plt.cm.get_cmap('viridis', len(df.columns) - 1)

# Plot all columns as bar plots
fig, axs = plt.subplots(len(df.columns) - 1, figsize=(10, 12))

# Loop through columns (excluding the 'index' column)
for i, col in enumerate(df.columns[1:]):
    # Sort the current column in ascending order
    sorted_df = df.sort_values(by=col, ascending=True)
    axs[i].bar(range(len(sorted_df)), sorted_df[col].tolist(),color=colormap(i))
    axs[i].set_title(col)
    axs[i].set_ylabel('Score')

    # TODO : Display the correct index sorted_df.index as x label 
    axs[i].set_xticks(range(len(sorted_df)))
    axs[i].set_xticklabels(sorted_df.index)

# Set font size for titles, labels, and tick labels
font_size = 4  # Adjust the font size as needed

for ax in axs:
    ax.title.set_fontsize(font_size)  # Title font size
    ax.xaxis.label.set_fontsize(font_size)  # X-axis label font size
    ax.yaxis.label.set_fontsize(font_size)  # Y-axis label font size
    ax.tick_params(axis='both', labelsize=font_size)  # Tick label font size

# Adjust layout
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('sorted_validation_plot.png', dpi=300, bbox_inches='tight')

# Show the plots
plt.show()

