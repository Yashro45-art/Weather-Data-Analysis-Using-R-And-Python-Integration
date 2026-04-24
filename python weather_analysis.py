import pandas as pd
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("===== WEATHER DATA ANALYSIS USING REAL DATASET =====\n")

# -------------------------
# Load Dataset from R
# -------------------------

ro.r('data(airquality)')
total_before = ro.r('nrow(airquality)')[0]

ro.r('airquality <- na.omit(airquality)')
total_after = ro.r('nrow(airquality)')[0]

missing_removed_percent = ((total_before - total_after) / total_before) * 100
print(f"Data Cleaning: {missing_removed_percent:.2f}% rows removed\n")

# Convert R dataframe to Python dataframe
with localconverter(ro.default_converter + pandas2ri.converter):
    df = ro.conversion.rpy2py(ro.r('airquality'))

print("Dataset (First 5 Rows):\n")
print(df.head())

# -------------------------
# Statistical Analysis (Python)
# -------------------------

print("\n--- Python Statistical Summary ---")
mean_temp = df["Temp"].mean()
print("Average Temperature:", mean_temp)
print("Maximum Temperature:", df["Temp"].max())
print("Minimum Temperature:", df["Temp"].min())

std_temp = df["Temp"].std()
print("Standard Deviation:", std_temp)

variation_percent = (std_temp / mean_temp) * 100
print(f"Temperature Variation: {variation_percent:.2f}%")

print("\nCorrelation Matrix:\n")
corr = df.corr()
print(corr)

temp_solar_corr = corr.loc["Temp", "Solar.R"] * 100
temp_wind_corr = corr.loc["Temp", "Wind"] * 100

print(f"\nTemp vs Solar Radiation Correlation: {temp_solar_corr:.2f}%")
print(f"Temp vs Wind Correlation: {temp_wind_corr:.2f}%")

# -------------------------
# GRAPH 1: Temperature Trend
# -------------------------

plt.figure()
plt.plot(df["Temp"])
plt.title("Daily Temperature Trend")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.savefig("temperature_trend.png")
plt.close()

# -------------------------
# GRAPH 2: Histogram
# -------------------------

plt.figure()
plt.hist(df["Temp"], bins=10)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.savefig("temperature_histogram.png")
plt.close()

# -------------------------
# GRAPH 3: Wind vs Temperature
# -------------------------

plt.figure()
plt.scatter(df["Wind"], df["Temp"])
plt.title("Wind vs Temperature")
plt.xlabel("Wind Speed")
plt.ylabel("Temperature")
plt.savefig("wind_vs_temp.png")
plt.close()

# -------------------------
# GRAPH 4: Solar Radiation vs Temperature
# -------------------------

plt.figure()
plt.scatter(df["Solar.R"], df["Temp"])
plt.title("Solar Radiation vs Temperature")
plt.xlabel("Solar Radiation")
plt.ylabel("Temperature")
plt.savefig("solar_vs_temp.png")
plt.close()

# -------------------------
# GRAPH 5: Boxplot
# -------------------------

plt.figure()
plt.boxplot(df["Temp"])
plt.title("Temperature Boxplot")
plt.ylabel("Temperature")
plt.savefig("temperature_boxplot.png")
plt.close()

print("\nAll graphs saved successfully!")

# -------------------------
# MACHINE LEARNING MODEL
# -------------------------

print("\n--- Machine Learning Model (Linear Regression) ---")

# Features
X = df[["Solar.R", "Wind"]]

# Target
y = df["Temp"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score (Accuracy):", r2)

# Graph: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.savefig("ml_prediction.png")
plt.close()

print("ML graph saved successfully!")

# -------------------------
# R Statistical Analysis
# -------------------------

print("\n--- Running R Statistical Analysis ---")

ro.r('''
mean_temp <- mean(airquality$Temp)
sd_temp <- sd(airquality$Temp)
cat("R Mean Temperature:", mean_temp, "\\n")
cat("R Standard Deviation:", sd_temp, "\\n")
''')

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")