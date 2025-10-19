
# --- Import data ---
exp1_path =  "/Users/elisavanderplas/Desktop/vdPlas_2025_Politics/exp1/behavioural_exp1.csv"
exp2_path =  "/Users/elisavanderplas/Desktop/vdPlas_2025_Politics/exp2/behavioural_exp2.csv"
exp2_demo_path =  "/Users/elisavanderplas/Desktop/vdPlas_2025_Politics/exp2/subjectLog_exp2.csv"
data_exp1 = pd.read_csv(exp1_path)
data_exp2 = pd.read_csv(exp2_path)
demo_exp2 = pd.read_csv(exp2_demo_path) 

# --- Prep data Exp 1 --- 
data_exp1['Video'] = pd.Categorical(data_exp1['Video'], categories = [1,2,3,4,5,6,7,8,9], ordered=False)
data_exp1['Video'] = data_exp1['Video'].cat.rename_categories(["immigration_threat","immigration_blame","immigration_uncertain", "environment_threat","environment_blame","environment_uncertain", "healthcare_threat","healthcare_blame","healthcare_uncertain"])

data_exp1['topic'] = pd.Categorical(data_exp1['video_2'], categories = [1,2,3], ordered=False)
data_exp1['topic'] = data_exp1['topic'].cat.rename_categories(["immigration","environment","healthcare"])

data_exp1['frame'] = pd.Categorical(data_exp1['video_3'], categories = [1,2,3], ordered=False)
data_exp1['frame'] = data_exp1['frame'].cat.rename_categories(["threat","blame","uncertain"])

data_exp1['gender'] = data_exp1['geslacht'] - 1 #back to 0 and 1 with 0 is male
data_exp1['online_use'] = data_exp1['V1_6'] #1 = Uses Social Media (such as Facebook, Twitter or Youtube) to follow the news
data_exp1['paper_read'] = data_exp1['V2a'] #1 = Reads political news in newspapers
data_exp1['online_read'] = data_exp1['V2b'] #1 = Reads political news o social media
data_exp1['interest_politics'] = data_exp1['V3'] #1-4 Likert from "Very interested in political issues" to "Not interested at all"
data_exp1['pre_CDA'] = data_exp1['V5_1']
data_exp1['pre_D66'] = data_exp1['V5_2']
#data_exp1['pre_GL'] = data_exp1['V5_3']
data_exp1['pre_PvdA'] = data_exp1['V5_4']
#data_exp1['pre_PVV'] = data_exp1['V5_5']
#data_exp1['pre_SP'] = data_exp1['V5_6']
data_exp1['pre_VVD'] = data_exp1['V5_7']
data_exp1['pre_50P'] = data_exp1['V5_8']
#data_exp1['leftright'] = data_exp1['V6']

data_exp1['importance_gezondheidsz'] = data_exp1['V13_1']
data_exp1['importance_immigr'] = data_exp1['V13_2']
data_exp1['importance_milieu'] = data_exp1['V13_3']

data_exp1['video_agreement'] = data_exp1['V10_4']
data_exp1['video_realiability'] =data_exp1['V10_5']
data_exp1['video_sharing'] = data_exp1['V10_6']

# Drop columns that start with 'V'
cols_to_drop = [col for col in data_exp1.columns if col.startswith('V')]
df_exp1 = data_exp1.drop(columns=cols_to_drop)

# Identify columns that start with 'INDEX'
index_cols = [col for col in df_exp1.columns if col.startswith(('INDEX', 'BIG', 'online', 'geboortejaar', 'paper'))]

# Clean and replace
for col in index_cols:
    # Convert 'NULL#' to NaN
    df_exp1[col] = df_exp1[col].replace('NULL#', np.nan)
    
    # Convert column to numeric (in case it's still object type)
    df_exp1[col] = pd.to_numeric(df_exp1[col], errors='coerce')
    
    # Compute median (ignoring NaN)
    median_value = df_exp1[col].median()
    
    # Fill NaN with median and convert to int
    df_exp1[col] = df_exp1[col].fillna(median_value).astype(float)

df_exp1['leeftijd'] = 2025 - df_exp1['geboortejaar']

## Anger prototype

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# First, let's create a correlation heatmap for anger with all demographics
demographic_vars = ['leeftijd', 'leftright', 'BIG5_O', 'BIG5_C', 'BIG5_E', 'BIG5_A', 'BIG5_N', 
                   'interest_politics', 'online_read',  'opleiding', 'gender', 'paper_read',
                   'INDEX_CYNICISM', 'INDEX_AUTHO', 'ANGER']

# Calculate correlations
correlation_matrix = df_exp1[demographic_vars].corr()
anger_correlations = correlation_matrix['ANGER'].drop('ANGER').sort_values(ascending=True)

# Create the "Recipe for Anger" heatmap
fig_heatmap = go.Figure(data=go.Heatmap(
    z=[anger_correlations.values],
    x=anger_correlations.index,
    y=['Correlation with Anger'],
    colorscale='RdBu_r',
    zmid=0,
    text=[[f'{val:.3f}' for val in anger_correlations.values]],
    texttemplate="%{text}",
    textfont={"size": 14},
    hoverinfo="x+z",
    colorbar=dict(title="Correlation")
))

fig_heatmap.update_layout(
    title='Recipe for Anger: Correlation Heatmap',
    xaxis_title='Demographic Variables',
    yaxis_title='',
    height=400,
    width=800
)

fig_heatmap.show()

# Now let's create a more detailed prototype analysis
def create_anger_prototype(df):
    """Create an anger prototype score and visualize"""
    
    # Standardize all variables
    scaler = StandardScaler()
    demo_scaled = scaler.fit_transform(df[demographic_vars[:-1]])  # Exclude ANGER
    demo_scaled_df = pd.DataFrame(demo_scaled, columns=demographic_vars[:-1])
    
    # Calculate prototype score (weighted by correlation with anger)
    anger_weights = anger_correlations.abs() / anger_correlations.abs().sum()
    df['anger_prototype_score'] = (demo_scaled_df * anger_weights).sum(axis=1)
    
    return df, anger_weights

# Apply the prototype analysis
df, anger_weights = create_anger_prototype(df_exp1)

# Visualize the top contributors to anger prototype
fig_weights = px.bar(x=anger_weights.sort_values(ascending=False).index,
                    y=anger_weights.sort_values(ascending=False).values,
                    title='Recipe for Anger: Variable Weights in Prototype',
                    labels={'x': 'Variables', 'y': 'Weight in Prototype Score'},
                    color=anger_weights.sort_values(ascending=False).values,
                    color_continuous_scale='Reds')

fig_weights.update_layout(xaxis_tickangle=-45)
fig_weights.show()

# Create demographic profiles for high vs low anger groups
df['anger_quartile'] = pd.qcut(df_exp1['ANGER'], 4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])

# Function to plot demographic comparisons
def plot_demographic_comparison(variable, title):
    fig = px.box(df_exp1, x='anger_quartile', y=variable,
                title=f'{title} by Anger Quartile',
                color='anger_quartile',
                color_discrete_sequence=px.colors.sequential.Reds)
    return fig

# Plot key demographics across anger quartiles
plot_demographic_comparison('leeftijd', 'Age')
plot_demographic_comparison('leftright', 'Political Affiliation (Left-Right)')
plot_demographic_comparison('BIG5_N', 'Neuroticism')
plot_demographic_comparison('INDEX_CYNICISM', 'Political Cynicism')
plot_demographic_comparison('opleiding', 'Education Level')

# Create a radar chart for the "angry prototype"
def create_anger_radar_chart(df):
    """Create a radar chart showing the angry prototype profile"""
    
    # Calculate mean values for highest anger quartile
    high_anger = df[df['anger_quartile'] == 'Q4 (Highest)']
    low_anger = df[df['anger_quartile'] == 'Q1 (Lowest)']
    
    # Select key variables for radar chart
    radar_vars = ['BIG5_N', 'INDEX_CYNICISM', 'leftright', 'leeftijd', 'interest_politics']
    
    high_means = high_anger[radar_vars].mean()
    low_means = low_anger[radar_vars].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=high_means.values,
        theta=radar_vars,
        fill='toself',
        name='High Anger Prototype',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=low_means.values,
        theta=radar_vars,
        fill='toself',
        name='Low Anger Prototype',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(high_means), max(low_means)) * 1.1]
            )),
        showlegend=True,
        title='Angry vs Calm Prototype: Demographic Profiles'
    )
    
    return fig

radar_chart = create_anger_radar_chart(df_exp1)
radar_chart.show()

# Create a summary table of the "angry prototype"
def print_anger_prototype_summary(df):
    high_anger = df[df['anger_quartile'] == 'Q4 (Highest)']
    
    print("=== THE ANGRY PROTOTYPE ===")
    print(f"Sample size: {len(high_anger)} participants")
    print("\nDemographic Profile:")
    print(f"• Average Age: {high_anger['leeftijd'].mean():.1f} years")
    print(f"• Political Leaning: {high_anger['leftright'].mean():.2f} (lower = more left-wing)")
    print(f"• Neuroticism: {high_anger['BIG5_N'].mean():.2f}")
    print(f"• Political Cynicism: {high_anger['INDEX_CYNICISM'].mean():.2f}")
    print(f"• Education Level: {high_anger['opleiding'].mean():.2f}")
    print(f"• Interest in Politics: {high_anger['interest_politics'].mean():.2f}")
    
    # Gender distribution
    if 'gender' in df.columns:
        gender_dist = high_anger['gender'].value_counts(normalize=True)
        print(f"• Gender Distribution: {gender_dist.to_dict()}")

print_anger_prototype_summary(df)

# Additional: Create a scatter plot matrix for top correlates
top_correlates = anger_correlations.abs().nlargest(6).index.tolist()
if 'ANGER' not in top_correlates:
    top_correlates.append('ANGER')

fig_scatter_matrix = px.scatter_matrix(df[top_correlates],
                                     dimensions=top_correlates,
                                     color=df['anger_quartile'],
                                     title='Scatter Matrix: Top Correlates with Anger',
                                     opacity=0.6)
fig_scatter_matrix.update_layout(height=800, width=800)
fig_scatter_matrix.show()