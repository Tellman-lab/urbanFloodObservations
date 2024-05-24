import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Load the data
data_path = r'Z:\\Projects\\NASA\\THP\\Data\\experimentsDataPaper\\trainingDataset0207\\perLabelSpec.csv'
data = pd.read_csv(data_path)

# Create a 'Region' column in the DataFrame
data['Region'] = data['Filename'].str[:3]

# Filter the data to only include 'Region' values with more than 4 rows
counts = data['Region'].value_counts()
data = data[data['Region'].isin(counts[counts > 4].index)]

# Calculate mean IoU (Specificity) per 'Region' and normalize these values
mean_iou_per_region = data.groupby('Region')['Specificity'].mean()
normalized_iou = (mean_iou_per_region - mean_iou_per_region.min()) / (mean_iou_per_region.max() - mean_iou_per_region.min())

# Map normalized IoU values to a blue color gradient
colors = plt.cm.Blues(normalized_iou)

# Create a dictionary to map 'Region' to its corresponding color
region_to_color = {region: mcolors.to_hex(color) for region, color in zip(mean_iou_per_region.index, colors)}

# Convert normalized_iou to DataFrame for CSV export
normalized_iou_df = normalized_iou.reset_index()
normalized_iou_df.columns = ['Region', 'Normalized_Specificity']

# Map colors to each region in the new DataFrame
normalized_iou_df['Color'] = normalized_iou_df['Region'].map(region_to_color)

# Save the DataFrame to a CSV file
output_path = 'Z:\\Projects\\NASA\\THP\\Data\\experimentsDataPaper\\trainingDataset0207\\normalized_iou_per_region.csv'
normalized_iou_df.to_csv(output_path, index=False)
print(f"CSV file saved to {output_path}")

# Visualization
# Increase global font sizes using rcParams for Matplotlib
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10})

# Set the style, color palette, and increase font size using Seaborn
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)

# Create a figure and axis for the boxplot with adjusted size
fig, ax = plt.subplots(figsize=(14, 8))

# Generate the boxplot with custom colors
sns.boxplot(x='Region', y='Specificity', data=data, ax=ax, showfliers=False, palette=region_to_color)

# Overlay individual data points on the boxplot
sns.stripplot(x='Region', y='Specificity', data=data, ax=ax, color='black', alpha=0.6, size=4, jitter=0.2)

# Set the labels
ax.set_ylabel('Specificity')

# Rotate the x-tick labels
plt.xticks(rotation=45)

# Create a colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=mean_iou_per_region.min(), vmax=mean_iou_per_region.max()))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Mean Specificity')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
