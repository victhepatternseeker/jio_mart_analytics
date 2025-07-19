# JioMart Sales Analytics Dashboard

This directory contains the Power BI dashboard for JioMart sales analytics. The dashboard provides comprehensive insights into sales performance, product mix, inventory management, and regional performance.

## Dashboard Components

1. **Sales Overview**
   - Total Revenue
   - Total Profit
   - Units Sold
   - Year-over-Year Growth

2. **Sales Trends**
   - Monthly Revenue Trend
   - Quarterly Sales Comparison
   - Day of Week Analysis

3. **Product Analysis**
   - Top Products by Revenue
   - Category Performance
   - Product Mix Distribution
   - Revenue by Product Level

4. **Inventory Management**
   - Reorder Point Analysis
   - Stock Level Monitoring
   - Safety Stock Levels

5. **Regional Performance**
   - City-wise Revenue
   - Profit Margin by City
   - Store Efficiency
   - Geographic Distribution

## Setup Instructions

1. First, prepare the data by running:
   ```bash
   python prepare_powerbi_data.py
   ```

2. Open Power BI Desktop and create a new report

3. Import the data from `sales_data_powerbi.xlsx`

4. Create the following visualizations:
   - KPI cards for key metrics
   - Line charts for trends
   - Bar charts for product and category analysis
   - Maps for geographic distribution
   - Heatmaps for inventory analysis
   - Matrix visualizations for detailed breakdowns

5. Add slicers for:
   - Date range
   - City
   - Category
   - Product

## Best Practices

- Use consistent color schemes
- Maintain proper data hierarchy
- Include drill-down capabilities
- Add tooltips for additional context
- Implement conditional formatting
- Use bookmarks for different views
- Add page navigation

## Maintenance

- Update the data source periodically
- Review and update visualizations
- Monitor performance
- Add new metrics as needed

## Security

- Set appropriate data access permissions
- Implement row-level security
- Protect sensitive information
- Regular backups of the dashboard

## Support

For any issues or enhancements, please refer to the main project documentation.
