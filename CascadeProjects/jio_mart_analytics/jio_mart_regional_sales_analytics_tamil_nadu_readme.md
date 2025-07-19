# 🛍️ JioMart Regional Sales Analytics – Tamil Nadu

## 🚀 Project Overview

This end-to-end data analytics project focuses on improving sales performance, product mix, and operational efficiency across 12 JioMart regional stores in Tamil Nadu. Using real-world sales simulation data, forecasting models, and BI techniques, we’ve generated deep business insights, optimized inventory, and delivered a professional executive report.

---

## 🛠️ Tools & Technologies

- **Languages & Libraries:** Python (pandas, numpy, matplotlib, seaborn, plotly, geopandas)
- **Forecasting Models:** ARIMA, SARIMA, Prophet
- **ML/Optimization:** scikit-learn, xgboost, optuna, scikit-optimize
- **Reporting & EDA:** pandas-profiling, reportlab
- **Geospatial Analysis:** folium
- **Notebook Environment:** Jupyter via Windsurf

---

## 📁 Folder Structure

```bash
jio_mart_analytics/
├── data/                      # Raw and processed data (CSV)
├── notebooks/                # Jupyter notebooks
├── reports/
│   ├── figures/              # All visualizations
│   └── JioMart_Regional_Sales_Insights_Executive_Brief.pdf
├── dashboards/               # Power BI .pbix files (optional)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 📈 Key Findings & Visualizations

### 🔹 Product Mix Optimization

- ✅ **High-ROI Categories:** Snacks, Personal Care, Pet Care
- 📉 **Low-Performing SKUs:** Recommended for replacement based on revenue per unit

### 🔹 Regional Strategy

- 🌟 **Premium Focus:** Chennai, Vellore, Tiruchirappalli (high profit margins)
- ⚙️ **Ops Optimization:** Chennai, Madurai, Thanjavur (low operational efficiency)

### 🔹 Inventory Recommendations

- 📦 **Reorder Points:** Calculated using sales velocity & lead time
- 🛡️ **Safety Stock:** Modeled for top SKUs by city & category
- 🔁 **Reorder Quantity Optimization:** Improved restocking logic

---

## 📊 Visualizations

All visuals are available in the `/reports/figures/` directory.

| Visualization | Description                      |
| ------------- | -------------------------------- |
|               | Revenue trends across cities     |
|               | Top 10 products by revenue       |
|               | Category-level margin comparison |
|               | Revenue/unit for lowest 10 SKUs  |
|               | Monthly profit by city           |
|               | Category vs City revenue heatmap |
|               | Forecasted revenue trend         |
|               | Inventory reorder heatmap        |

---

## 📍 Executive Report

📄 [Download Full Report (PDF)](reports/JioMart_Regional_Sales_Insights_Executive_Brief.pdf)\
Includes: project intro, key metrics, visuals, inventory strategy, business insights.

---

## 🧰 Environment & Dependencies

To reproduce this project:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- pandas==2.1.0, numpy==1.24.3, scipy==1.10.1
- scikit-learn==1.3.0, statsmodels==0.14.0, prophet==1.1.4
- matplotlib==3.7.2, seaborn==0.12.2, plotly==5.17.0
- xgboost==1.7.6, optuna==3.3.0, scikit-optimize==0.9.0
- geopandas==0.14.0, folium==0.15.0
- openpyxl==3.1.2, reportlab==4.0.9, pandas-profiling==3.6.6

---

## 📌 Summary

This project showcases how a data analyst can leverage sales data, forecasting models, and BI strategy to make data-backed decisions for real retail challenges. From SKU pruning to premium targeting to inventory control — this is the kind of analysis businesses need today.

**Built with 💼 by Vignesh Ramachandran**

