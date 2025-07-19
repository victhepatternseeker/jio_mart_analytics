import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from scipy.stats import zscore
from datetime import datetime
import os
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation

# Add city coordinates for Tamil Nadu cities
CITY_COORDS = {
    'Chennai': (13.0827, 80.2707),
    'Coimbatore': (11.0168, 76.9553),
    'Madurai': (9.9312, 78.1173),
    'Salem': (11.6700, 78.1400),
    'Trichy': (10.8017, 78.6872),
    'Tirunelveli': (8.7139, 77.7108),
    'Erode': (11.3451, 77.7347),
    'Thanjavur': (10.7905, 79.1423),
    'Vellore': (12.9295, 79.1578),
    'Tiruchirappalli': (10.7905, 78.6872),
    'Kancheepuram': (12.8333, 79.7000),
    'Dindigul': (10.0000, 78.1167)
}

# Constants
CITIES = ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Trichy', 'Tirunelveli',
          'Erode', 'Thanjavur', 'Vellore', 'Tiruchirappalli', 'Kancheepuram', 'Dindigul']

# Create output directories if they don't exist
os.makedirs('reports/figures', exist_ok=True)

class SalesAnalyzer:
    def __init__(self, data_path):
        # Get the absolute path of the data file
        self.data_path = os.path.abspath(data_path)
        self.df = self.load_data()
        
    def load_data(self):
        """Load and preprocess the sales data"""
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def analyze_sales_trends(self):
        """Analyze sales trends and patterns"""
        print("\n=== Sales Trends Analysis ===")
        
        # Monthly sales by city
        monthly_sales = self.df.groupby(['City', pd.Grouper(key='Date', freq='M')])['Revenue'].sum()
        monthly_sales = monthly_sales.unstack('City')
        
        # Plot monthly sales trends
        plt.figure(figsize=(15, 8))
        for city in CITIES:
            if city in monthly_sales.columns:
                plt.plot(monthly_sales.index, monthly_sales[city], label=city)
        plt.title('Monthly Sales Trends by City')
        plt.xlabel('Date')
        plt.ylabel('Revenue')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('reports/figures/monthly_sales_trends.png')
        
        # Create heatmap of revenue by category and city
        heatmap_data = self.df.groupby(['Category', 'City'])['Revenue'].sum().unstack()
        plt.figure(figsize=(15, 10))
        sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.0f')
        plt.title('Revenue Heatmap by Category and City')
        plt.tight_layout()
        plt.savefig('reports/figures/revenue_heatmap.png')
        plt.close()
        
        # Create geomap of cities
        m = folium.Map(location=[11, 78], zoom_start=7)
        
        # Add markers for each city with revenue info
        for city in CITIES:
            coords = CITY_COORDS[city]
            city_revenue = monthly_sales[city].sum()
            folium.Marker(
                location=coords,
                popup=f'{city}: {city_revenue:,.0f} INR',
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)
        
        # Save the map
        m.save('reports/figures/city_map.html')
        
        return monthly_sales

    def optimize_product_mix(self):
        """Perform product mix optimization"""
        print("\n=== Product Mix Optimization ===")
        
        # Calculate product performance metrics
        product_perf = self.df.groupby(['Product_Name', 'Category']).agg({
            'Units_Sold': 'sum',
            'Revenue': 'sum',
            'Profit_Margin': 'mean'
        }).reset_index()
        
        # Calculate ROI
        product_perf['ROI'] = (product_perf['Revenue'] * product_perf['Profit_Margin']) / product_perf['Units_Sold']
        
        # Top 10 products by revenue
        top_products = product_perf.sort_values('Revenue', ascending=False).head(10)
        plt.figure(figsize=(15, 8))
        sns.barplot(data=top_products, x='Revenue', y='Product_Name', hue='Category')
        plt.title('Top 10 Products by Revenue')
        plt.tight_layout()
        plt.savefig('reports/figures/top_products.png')
        plt.close()
        
        # Profit margin comparison by category
        category_profit = self.df.groupby('Category').agg({
            'Revenue': 'sum',
            'Profit_Margin': 'mean'
        }).reset_index()
        plt.figure(figsize=(15, 8))
        sns.barplot(data=category_profit, x='Category', y='Profit_Margin')
        plt.title('Profit Margin Comparison by Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/figures/profit_margin_by_category.png')
        plt.close()
        
        # Low-performing SKUs
        product_perf['Revenue_per_Unit'] = product_perf['Revenue'] / product_perf['Units_Sold']
        low_performing = product_perf.sort_values('Revenue_per_Unit').head(10)
        plt.figure(figsize=(15, 8))
        sns.barplot(data=low_performing, x='Revenue_per_Unit', y='Product_Name', hue='Category')
        plt.title('Low-Performing SKUs (Bottom 10 by Revenue per Unit Sold)')
        plt.tight_layout()
        plt.savefig('reports/figures/low_performing_skus.png')
        plt.close()
        
        # Category-wise optimization
        category_optimization = []
        for category in self.df['Category'].unique():
            cat_data = product_perf[product_perf['Category'] == category]
            
            # Calculate category metrics
            cat_metrics = {
                'Category': category,
                'Total_Revenue': cat_data['Revenue'].sum(),
                'Average_ROI': cat_data['ROI'].mean(),
                'Underperforming_Count': (cat_data['ROI'] < cat_data['ROI'].quantile(0.25)).sum(),
                'High_ROI_Count': (cat_data['ROI'] > cat_data['ROI'].quantile(0.75)).sum()
            }
            category_optimization.append(cat_metrics)
        
        category_df = pd.DataFrame(category_optimization)
        
        # Generate recommendations
        recommendations = []
        for _, row in category_df.iterrows():
            if row['Underperforming_Count'] / len(self.df[self.df['Category'] == row['Category']]) > 0.3:
                recommendations.append({
                    'Category': row['Category'],
                    'Recommendation': 'Reduce SKU count',
                    'Reason': 'High number of underperforming SKUs'
                })
            if row['Average_ROI'] > category_df['Average_ROI'].quantile(0.75):
                recommendations.append({
                    'Category': row['Category'],
                    'Recommendation': 'Increase SKU count',
                    'Reason': 'High average ROI'
                })
        
        return category_df, recommendations

    def forecast_sales(self):
        """Forecast sales using ARIMA and Prophet"""
        print("\n=== Sales Forecasting ===")
        
        # Aggregate data for forecasting
        monthly_sales = self.df.groupby(self.df['Date'].dt.to_period('M'))['Revenue'].sum()
        
        # Split data
        train = monthly_sales.iloc[:-12]  # Last 12 months as test
        test = monthly_sales.iloc[-12:]
        
        # ARIMA Model
        print("\nARIMA Model Results:")
        best_aic = float('inf')
        best_order = None
        
        # Try different ARIMA parameters
        try:
            # Try (1,1,1) first as it's a common starting point
            model = ARIMA(train, order=(1,1,1))
            model_fit = model.fit()
            best_order = (1,1,1)
            print("Using ARIMA(1,1,1) parameters")
        except:
            # If that fails, try (0,1,0) which is a simpler model
            try:
                model = ARIMA(train, order=(0,1,0))
                model_fit = model.fit()
                best_order = (0,1,0)
                print("Using ARIMA(0,1,0) parameters")
            except:
                print("Warning: ARIMA model failed to converge. Using simple moving average instead.")
                # If ARIMA fails completely, use a simple moving average as fallback
                forecast = train.rolling(window=3).mean()[-12:].values
                return forecast
        
        print(f"Using ARIMA parameters: {best_order}")
        
        # Forecast
        forecast = model_fit.forecast(steps=12)
        model_fit = model.fit()
        
        # Forecast
        forecast = model_fit.forecast(steps=12)
        
        # Calculate metrics
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        
        print(f"\nARIMA Forecast Performance:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Plot ARIMA results
        plt.figure(figsize=(12, 6))
        plt.plot(train.index.to_timestamp(), train.values, label='Train')
        plt.plot(test.index.to_timestamp(), test.values, label='Test')
        plt.plot(test.index.to_timestamp(), forecast, label='Forecast')
        plt.title('ARIMA Sales Forecast')
        plt.legend()
        plt.savefig('reports/figures/arima_forecast.png')
        plt.close()
        
        # Prophet Model
        print("\nProphet Model Results:")
        
        # Prepare data for Prophet
        prophet_data = monthly_sales.to_frame().reset_index()
        prophet_data.columns = ['ds', 'y']
        
        # Split data
        train_prophet = prophet_data.iloc[:-12]
        test_prophet = prophet_data.iloc[-12:]
        
        # Add holidays and seasonality
        holidays = pd.DataFrame({
            'holiday': 'diwali',
            'ds': pd.to_datetime(['2023-10-24', '2024-11-12']),
            'lower_window': 0,
            'upper_window': 1
        })
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            holidays=holidays,
            seasonality_mode='multiplicative'
        )
        model.fit(train_prophet)
        
        # Make predictions
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        
        # Calculate metrics
        prophet_mae = mean_absolute_error(
            test_prophet['y'],
            forecast.iloc[-12:]['yhat']
        )
        prophet_rmse = np.sqrt(mean_squared_error(
            test_prophet['y'],
            forecast.iloc[-12:]['yhat']
        ))
        
        print(f"\nProphet Forecast Performance:")
        print(f"MAE: {prophet_mae:.2f}")
        print(f"RMSE: {prophet_rmse:.2f}")
        
        # Plot Prophet results
        fig1 = model.plot(forecast)
        plt.savefig('reports/figures/prophet_forecast.png')
        plt.close()
        
        return forecast

    def generate_inventory_recommendations(self):
        """Generate inventory reordering recommendations"""
        print("\n=== Inventory Reordering Recommendations ===")
        
        # Calculate sales velocity (units sold per day)
        velocity = self.df.groupby(['City', 'Category', 'Product_Name'])['Units_Sold'].sum() / len(self.df['Date'].unique())
        
        # Calculate safety stock (2 weeks worth of sales)
        safety_stock = velocity * 14
        
        # Calculate reorder point (1 week lead time + safety stock)
        reorder_point = velocity * 7 + safety_stock
        
        # Get top 100 products by sales volume for recommendations
        top_products = velocity.sort_values(ascending=False).head(100)
        
        # Create recommendations dataframe
        recommendations = []
        for (city, category, product), vel in top_products.items():
            recommendations.append({
                'City': city,
                'Category': category,
                'Product': product,
                'Daily_Velocity': vel,
                'Safety_Stock': safety_stock[(city, category, product)],
                'Reorder_Point': reorder_point[(city, category, product)],
                'Reorder_Quantity': safety_stock[(city, category, product)] * 2  # Double safety stock as reorder quantity
            })
        
        # Create a pivot table for visualization
        rec_df = pd.DataFrame(recommendations)
        pivot = pd.pivot_table(rec_df, values='Reorder_Point', 
                             index='Category', 
                             columns='City',
                             aggfunc='mean')
        
        # Plot reorder point heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.0f')
        plt.title('Average Reorder Points by Category and City')
        plt.tight_layout()
        plt.savefig('reports/figures/reorder_points.png')
        plt.close()
        
        return rec_df

    def generate_regional_strategy(self):
        """Generate regional marketing and inventory strategies"""
        print("\n=== Regional Strategy Analysis ===")
        
        # Store-wise performance
        store_perf = self.df.groupby(['City', 'Store_ID']).agg({
            'Revenue': 'sum',
            'Profit_Margin': 'mean',
            'Units_Sold': 'sum'
        }).reset_index()
        
        # Calculate store efficiency
        store_perf['Efficiency'] = store_perf['Revenue'] / store_perf['Units_Sold']
        
        # City-wise performance metrics
        city_metrics = []
        for city in CITIES:
            city_data = store_perf[store_perf['City'] == city]
            metrics = {
                'City': city,
                'Total_Revenue': city_data['Revenue'].sum(),
                'Average_Profit_Margin': city_data['Profit_Margin'].mean(),
                'Average_Efficiency': city_data['Efficiency'].mean(),
                'Store_Count': len(city_data)
            }
            city_metrics.append(metrics)
        
        city_df = pd.DataFrame(city_metrics)
        
        # Plot city-wise profit trend
        plt.figure(figsize=(15, 8))
        sns.lineplot(data=city_df, x='City', y='Average_Profit_Margin', marker='o')
        plt.title('City-wise Profit Margin Trend')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('reports/figures/city_profit_trend.png')
        plt.close()
        
        # Generate regional recommendations
        recommendations = []
        for _, row in city_df.iterrows():
            if row['Average_Profit_Margin'] > city_df['Average_Profit_Margin'].quantile(0.75):
                recommendations.append({
                    'City': row['City'],
                    'Strategy': 'Premium Product Focus',
                    'Reason': 'High average profit margin'
                })
            if row['Store_Count'] < 2:
                recommendations.append({
                    'City': row['City'],
                    'Strategy': 'Store Expansion',
                    'Reason': 'Limited store presence'
                })
            if row['Average_Efficiency'] < city_df['Average_Efficiency'].quantile(0.25):
                recommendations.append({
                    'City': row['City'],
                    'Strategy': 'Operational Optimization',
                    'Reason': 'Low operational efficiency'
                })
        
        return city_df, recommendations

    def generate_executive_summary(self):
        """Generate executive summary PDF"""
        print("\n=== Generating Executive Summary PDF ===")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            "reports/JioMart_Regional_Sales_Insights_Executive_Brief.pdf",
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20
        )
        normal_style = styles['Normal']
        
        # Create content
        content = []
        
        # Title
        content.append(Paragraph("JioMart Regional Sales Insights", title_style))
        content.append(Paragraph("Executive Brief", heading_style))
        content.append(Spacer(1, 20))
        
        # Key Findings
        content.append(Paragraph("Key Findings:", heading_style))
        content.append(Paragraph("- High-performing categories: Snacks, Personal Care, Pet Care", normal_style))
        content.append(Paragraph("- Cities with premium product potential: Chennai, Vellore, Tiruchirappalli", normal_style))
        content.append(Paragraph("- Operational optimization needed in: Chennai, Madurai, Thanjavur", normal_style))
        content.append(Spacer(1, 20))
        
        # Add key visualizations
        def add_image(path, width=400, height=300):
            img = Image(path, width=width, height=height)
            content.append(img)
            content.append(Spacer(1, 20))
        
        add_image('reports/figures/top_products.png')
        add_image('reports/figures/profit_margin_by_category.png')
        add_image('reports/figures/revenue_heatmap.png')
        add_image('reports/figures/city_profit_trend.png')
        add_image('reports/figures/reorder_points.png')
        
        # Save PDF
        doc.build(content)
        print("Executive summary PDF generated successfully!")

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Sales Analysis Pipeline...")
        
        # 1. Sales Trends Analysis
        monthly_sales = self.analyze_sales_trends()
        
        # 2. Product Mix Optimization
        category_df, product_recs = self.optimize_product_mix()
        print("\nProduct Mix Recommendations:")
        for rec in product_recs:
            print(f"Category: {rec['Category']}, Recommendation: {rec['Recommendation']}, Reason: {rec['Reason']}")
        
        # 3. Sales Forecasting
        forecast = self.forecast_sales()
        
        # 4. Inventory Recommendations
        inventory_recs = self.generate_inventory_recommendations()
        print("\nInventory Reordering Recommendations:")
        print(inventory_recs.head())
        
        # 5. Regional Strategy
        city_df, regional_recs = self.generate_regional_strategy()
        print("\nRegional Strategy Recommendations:")
        for rec in regional_recs:
            print(f"City: {rec['City']}, Strategy: {rec['Strategy']}, Reason: {rec['Reason']}")
        
        # 6. Generate Executive Summary
        self.generate_executive_summary()
        
        print("\nAnalysis Complete! Reports saved in 'reports' directory.")

if __name__ == "__main__":
    # Use absolute path to the data file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sales_data.csv')
    analyzer = SalesAnalyzer(data_path)
    analyzer.run_analysis()
