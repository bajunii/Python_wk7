"""
Data Analysis Assignment - Week 7
Objective: Load and analyze a dataset using pandas and create visualizations with matplotlib

Author: Student
Date: September 7, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import requests
import io

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("DATA ANALYSIS ASSIGNMENT - WEEK 7")
print("=" * 60)

# ============================================================================
# TASK 1: LOAD AND EXPLORE THE DATASET
# ============================================================================

print("\n📊 TASK 1: LOADING AND EXPLORING THE DATASET")
print("-" * 50)

# Load the Iris dataset from online source
print("Loading the Iris dataset...")
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

try:
    # Load dataset
    df = pd.read_csv(url)
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    # Fallback: Create sample data if online loading fails
    print("Creating sample dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'sepal_length': np.random.normal(5.8, 0.8, 150),
        'sepal_width': np.random.normal(3.0, 0.4, 150),
        'petal_length': np.random.normal(3.8, 1.8, 150),
        'petal_width': np.random.normal(1.2, 0.8, 150),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
    })

print(f"\n📋 Dataset shape: {df.shape} (rows, columns)")

# Display first few rows
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\n📊 Dataset Information:")
print(df.info())

print("\n📈 Data Types:")
print(df.dtypes)

# Check for missing values
print("\n🔍 Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("\n🧹 Cleaning missing values...")
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("✅ Missing values handled!")
else:
    print("✅ No missing values found!")

print("\n📊 Final dataset shape after cleaning:", df.shape)

# ============================================================================
# TASK 2: BASIC DATA ANALYSIS
# ============================================================================

print("\n\n📊 TASK 2: BASIC DATA ANALYSIS")
print("-" * 50)

# Basic statistics for numerical columns
print("\n📈 Basic Statistics for Numerical Columns:")
numerical_stats = df.describe()
print(numerical_stats)

# Group by species and compute mean
print("\n🔬 Analysis by Species (Grouping):")
species_analysis = df.groupby('species').agg({
    'sepal_length': ['mean', 'std', 'min', 'max'],
    'sepal_width': ['mean', 'std', 'min', 'max'],
    'petal_length': ['mean', 'std', 'min', 'max'],
    'petal_width': ['mean', 'std', 'min', 'max']
}).round(3)

print(species_analysis)

# Additional analysis - correlation matrix
print("\n🔗 Correlation Matrix:")
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
print(correlation_matrix.round(3))

# Findings and observations
print("\n\n🔍 KEY FINDINGS AND OBSERVATIONS:")
print("-" * 40)

# Find species with largest average petal length
max_petal_species = df.groupby('species')['petal_length'].mean().idxmax()
max_petal_value = df.groupby('species')['petal_length'].mean().max()

print(f"1. Species with largest average petal length: {max_petal_species} ({max_petal_value:.2f} cm)")

# Find strongest correlation
corr_vals = correlation_matrix.values
np.fill_diagonal(corr_vals, 0)  # Remove diagonal values
max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_vals)), corr_vals.shape)
strongest_corr = correlation_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
feature1 = correlation_matrix.index[max_corr_idx[0]]
feature2 = correlation_matrix.columns[max_corr_idx[1]]

print(f"2. Strongest correlation: {feature1} vs {feature2} (r = {strongest_corr:.3f})")

# Species distribution
species_counts = df['species'].value_counts()
print(f"3. Species distribution:")
for species, count in species_counts.items():
    print(f"   - {species}: {count} samples ({count/len(df)*100:.1f}%)")

# ============================================================================
# TASK 3: DATA VISUALIZATION
# ============================================================================

print("\n\n📊 TASK 3: DATA VISUALIZATION")
print("-" * 50)

# Create a figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Comprehensive Visualization', fontsize=16, fontweight='bold')

# 1. Line Chart - Create time series data for demonstration
print("1. Creating Line Chart (Trends over time)...")
dates = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
df_with_dates = df.copy()
df_with_dates['date'] = dates
daily_avg = df_with_dates.groupby('date')['petal_length'].mean()

ax1.plot(daily_avg.index, daily_avg.values, linewidth=2, color='blue', alpha=0.7)
ax1.set_title('Average Petal Length Trend Over Time', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Petal Length (cm)')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 2. Bar Chart - Average measurements by species
print("2. Creating Bar Chart (Comparison across categories)...")
species_means = df.groupby('species')[['sepal_length', 'petal_length']].mean()
x_pos = np.arange(len(species_means.index))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, species_means['sepal_length'], width, 
                label='Sepal Length', alpha=0.8, color='skyblue')
bars2 = ax2.bar(x_pos + width/2, species_means['petal_length'], width,
                label='Petal Length', alpha=0.8, color='lightcoral')

ax2.set_title('Average Sepal and Petal Length by Species', fontweight='bold')
ax2.set_xlabel('Species')
ax2.set_ylabel('Length (cm)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(species_means.index)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 3. Histogram - Distribution of sepal length
print("3. Creating Histogram (Distribution analysis)...")
ax3.hist(df['sepal_length'], bins=20, alpha=0.7, color='green', edgecolor='black')
ax3.axvline(df['sepal_length'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {df["sepal_length"].mean():.2f}')
ax3.axvline(df['sepal_length'].median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {df["sepal_length"].median():.2f}')
ax3.set_title('Distribution of Sepal Length', fontweight='bold')
ax3.set_xlabel('Sepal Length (cm)')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Scatter Plot - Relationship between two variables
print("4. Creating Scatter Plot (Relationship analysis)...")
colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    ax4.scatter(species_data['sepal_length'], species_data['petal_length'], 
                c=colors[species], label=species, alpha=0.7, s=50)

# Add trend line
z = np.polyfit(df['sepal_length'], df['petal_length'], 1)
p = np.poly1d(z)
ax4.plot(df['sepal_length'].sort_values(), p(df['sepal_length'].sort_values()), 
         "k--", alpha=0.8, linewidth=2, label='Trend Line')

ax4.set_title('Sepal Length vs Petal Length by Species', fontweight='bold')
ax4.set_xlabel('Sepal Length (cm)')
ax4.set_ylabel('Petal Length (cm)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('iris_analysis_plots.png', dpi=300, bbox_inches='tight')
print("📊 Plots saved as 'iris_analysis_plots.png'")

# Show the plots
plt.show()

# ============================================================================
# ADDITIONAL ANALYSIS - BONUS VISUALIZATIONS
# ============================================================================

print("\n\n🎯 BONUS: ADDITIONAL VISUALIZATIONS")
print("-" * 50)

# Create additional plots
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Advanced Iris Dataset Analysis', fontsize=16, fontweight='bold')

# 5. Box Plot - Distribution comparison
print("5. Creating Box Plot (Distribution comparison)...")
df_melted = pd.melt(df, id_vars=['species'], 
                    value_vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                    var_name='measurement', value_name='value')
sns.boxplot(data=df_melted, x='measurement', y='value', hue='species', ax=ax5)
ax5.set_title('Distribution of Measurements by Species', fontweight='bold')
ax5.set_xlabel('Measurement Type')
ax5.set_ylabel('Value (cm)')
ax5.tick_params(axis='x', rotation=45)

# 6. Correlation Heatmap
print("6. Creating Correlation Heatmap...")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6,
            square=True, fmt='.3f')
ax6.set_title('Correlation Matrix Heatmap', fontweight='bold')

# 7. Violin Plot - Density distribution
print("7. Creating Violin Plot...")
sns.violinplot(data=df, x='species', y='petal_width', ax=ax7)
ax7.set_title('Petal Width Distribution by Species', fontweight='bold')
ax7.set_xlabel('Species')
ax7.set_ylabel('Petal Width (cm)')

# 8. Pair Plot simulation (simplified)
print("8. Creating Feature Comparison Plot...")
ax8.scatter(df['sepal_width'], df['petal_width'], c=[colors[species] for species in df['species']], 
            alpha=0.7, s=50)
ax8.set_title('Sepal Width vs Petal Width', fontweight='bold')
ax8.set_xlabel('Sepal Width (cm)')
ax8.set_ylabel('Petal Width (cm)')
ax8.grid(True, alpha=0.3)

# Add legend for species colors
for species, color in colors.items():
    ax8.scatter([], [], c=color, label=species)
ax8.legend()

plt.tight_layout()
plt.savefig('iris_advanced_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Advanced plots saved as 'iris_advanced_analysis.png'")
plt.show()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n\n📋 FINAL SUMMARY REPORT")
print("=" * 60)

print(f"""
Dataset Overview:
- Total samples: {len(df)}
- Features: {len(df.columns)}
- Species: {len(df['species'].unique())}
- No missing values: {'✅' if df.isnull().sum().sum() == 0 else '❌'}

Key Statistical Insights:
- Average sepal length: {df['sepal_length'].mean():.2f} ± {df['sepal_length'].std():.2f} cm
- Average petal length: {df['petal_length'].mean():.2f} ± {df['petal_length'].std():.2f} cm
- Strongest correlation: {feature1} vs {feature2} (r = {strongest_corr:.3f})
- Most variable feature: {df.select_dtypes(include=[np.number]).std().idxmax()}

Species Characteristics:
""")

for species in df['species'].unique():
    species_data = df[df['species'] == species]
    print(f"- {species.capitalize()}:")
    print(f"  • Count: {len(species_data)} samples")
    print(f"  • Avg petal length: {species_data['petal_length'].mean():.2f} cm")
    print(f"  • Avg sepal length: {species_data['sepal_length'].mean():.2f} cm")

print(f"""
Visualizations Created:
✅ Line chart - Temporal trends
✅ Bar chart - Species comparison  
✅ Histogram - Distribution analysis
✅ Scatter plot - Feature relationships
✅ Box plot - Distribution comparison
✅ Heatmap - Correlation analysis
✅ Violin plot - Density visualization
✅ Feature comparison plot

Files Generated:
📁 iris_analysis_plots.png
📁 iris_advanced_analysis.png
📁 data_analysis_assignment.py

Analysis Complete! 🎉
""")

print("=" * 60)
print("Assignment completed successfully!")
print("All requirements have been fulfilled:")
print("✅ Data loading and exploration")
print("✅ Basic data analysis") 
print("✅ Data visualization (4+ different types)")
print("✅ Findings and observations")
print("=" * 60)
