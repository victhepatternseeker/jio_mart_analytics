import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Cities and stores
CITIES = ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Trichy', 'Tirunelveli',
          'Erode', 'Thanjavur', 'Vellore', 'Tiruchirappalli', 'Kancheepuram', 'Dindigul']
STORES_PER_CITY = 2

# Product categories
CATEGORIES = ['Fruits & Vegetables', 'Dairy & Eggs', 'Bakery', 'Beverages', 'Snacks', 'Personal Care', 
              'Home Care', 'Baby Care', 'Pet Care', 'Meat & Fish', 'Frozen Foods', 'Household Items']

# Generate date range (1 year)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')

# Create empty DataFrame
data = []

# Generate data for each store
for city in CITIES:
    for store in range(1, STORES_PER_CITY + 1):
        store_id = f"{city[:3]}_{store}"
        
        # Generate data for each day
        for date in dates:
            # Generate random sales data
            for category in CATEGORIES:
                units_sold = np.random.poisson(lam=50)  # Base sales
                
                # Add seasonal variation
                if category == 'Fruits & Vegetables':
                    units_sold += np.random.poisson(lam=20) if date.month in [6, 7, 8] else 0
                elif category == 'Bakery':
                    units_sold += np.random.poisson(lam=15) if date.weekday() in [5, 6] else 0
                
                # Add city-specific variation
                city_factor = random.uniform(0.8, 1.2)
                units_sold = int(units_sold * city_factor)
                
                # Generate revenue and profit margin
                avg_price = np.random.uniform(50, 200)
                revenue = units_sold * avg_price
                profit_margin = np.random.uniform(0.15, 0.35)
                
                # Generate product name
                product_name = f"{category.replace(' & ', '_')}_{random.randint(1, 100)}"
                
                data.append({
                    'Date': date,
                    'Store_ID': store_id,
                    'City': city,
                    'Product_Name': product_name,
                    'Category': category,
                    'Units_Sold': units_sold,
                    'Revenue': revenue,
                    'Profit_Margin': profit_margin
                })

# Create DataFrame
df = pd.DataFrame(data)

# Add some noise
np.random.seed(42)
df['Revenue'] += np.random.normal(0, 500, len(df))
df['Units_Sold'] += np.random.normal(0, 5, len(df))
df['Profit_Margin'] += np.random.normal(0, 0.02, len(df))

# Round values
df['Revenue'] = df['Revenue'].round(2)
df['Profit_Margin'] = df['Profit_Margin'].round(4)
df['Units_Sold'] = df['Units_Sold'].round(0).astype(int)

# Save to CSV
df.to_csv('sales_data.csv', index=False)
