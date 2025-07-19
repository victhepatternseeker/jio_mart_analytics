import pandas as pd
import numpy as np
from datetime import datetime
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the sales data
df = pd.read_csv(os.path.join(project_root, 'data', 'sales_data.csv'))

# Add date components
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Quarter'] = df['Date'].dt.quarter

# Create additional calculated metrics
df['Profit'] = df['Revenue'] * df['Profit_Margin']
df['Units_Per_Day'] = df['Units_Sold']
df['Revenue_Per_Unit'] = df['Revenue'] / df['Units_Sold']

def get_month_name(month_num):
    return datetime(2020, month_num, 1).strftime('%B')

def get_quarter_name(quarter_num):
    return f'Q{quarter_num}'

# Create lookup tables for better visualization
df['Month_Name'] = df['Month'].apply(get_month_name)
df['Quarter_Name'] = df['Quarter'].apply(get_quarter_name)

# Create category hierarchy
df['Product_Level'] = df['Category'] + ' - ' + df['Product_Name']

# Save the enhanced dataset
df.to_excel(os.path.join(project_root, 'dashboards', 'sales_data_powerbi.xlsx'), index=False)

print("Data prepared for Power BI. File saved as sales_data_powerbi.xlsx")
