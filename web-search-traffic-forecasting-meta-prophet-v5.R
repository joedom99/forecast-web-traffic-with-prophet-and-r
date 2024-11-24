# Web traffic forecasting with Meta Prophet library for R
# Version 5, 2024-11-24
# By: Joe Domaleski
# https://blog.marketingdatascience.ai

# Load necessary libraries
library(tidyverse)
library(lubridate)
library(prophet)
library(ggplot2)

# Step 1: Load and Inspect Data
# --------------------------------------
file_name <- "web-search-traffic-16mos.csv"
data <- read_csv(file_name)

# Inspect data
summary(data)

# Step 2: Detect and Handle Spikes
# --------------------------------------
threshold <- quantile(data$Clicks, 0.95)
data$IsSpike <- ifelse(data$Clicks > threshold, TRUE, FALSE)
spikes <- data[data$IsSpike == TRUE, ]
cat("Detected spikes with 95th percentile threshold:\n")
print(spikes)

# Create a holidays dataframe for detected spikes
holidays <- spikes %>%
  select(ds = Date) %>%
  mutate(
    holiday = "Spike",
    lower_window = 0,
    upper_window = ifelse(ds %in% as.Date(c("2023-11-07", "2024-11-05")), 2, 1)
  )

# Cap spikes at the threshold
data$Clicks <- ifelse(data$Clicks > threshold, threshold, data$Clicks)

# Step 3: Prepare Data for Prophet
# --------------------------------------
data <- data %>%
  mutate(Date = as.Date(Date, format = "%Y-%m-%d")) %>%
  rename(ds = Date, y = Clicks)

# Train-test split
train_data <- data[1:(nrow(data) - 60), ]
test_data <- data[(nrow(data) - 59):nrow(data), ]

# Step 4: Fit Prophet Model
# --------------------------------------
model <- prophet(
  daily.seasonality = FALSE,
  yearly.seasonality = TRUE,
  weekly.seasonality = TRUE,
  holidays = holidays
)

# Fit the model with training data
model <- fit.prophet(model, train_data)

# Step 5: Cross-Validation
# --------------------------------------
message("Performing cross-validation...")
cv_results <- cross_validation(
  model,
  horizon = 15,        # Forecast horizon: 30 days
  initial = 365,       # Initial training period: 12 months (365 days)
  period = 15,         # Rolling validation every 15 days
  units = "days"       # Specify the units
)

# Performance metrics
performance <- performance_metrics(cv_results)
print("Cross-Validation Performance Metrics:")
print(performance)

# Plot cross-validation RMSE
plot_cross_validation_metric(cv_results, metric = "rmse") +
  labs(title = "Cross-Validation RMSE", x = "Horizon (days)", y = "RMSE")

# Step 6: Forecast Future Traffic
# --------------------------------------
future <- make_future_dataframe(model, periods = 60)
forecast <- predict(model, future)

# Evaluate test set performance
message("Evaluating test set performance...")
actual <- test_data$y
predicted <- forecast$yhat[(nrow(forecast) - nrow(test_data) + 1):nrow(forecast)]

mae <- mean(abs(actual - predicted))
rmse <- sqrt(mean((actual - predicted)^2))
mape <- mean(abs((actual - predicted) / actual)) * 100

cat("Test Set Performance Metrics:\n")
cat("MAE:", mae, "\n")
cat("MAE Percentage:", mae / mean(actual) * 100, "%\n")
cat("RMSE:", rmse, "\n")
cat("RMSE Percentage:", rmse / mean(actual) * 100, "%\n")
cat("MAPE:", mape, "%\n")

# Visualize test set performance
# Actual vs. Predicted for Test Set
comparison <- data.frame(
  Date = test_data$ds,
  Actual = actual,
  Predicted = predicted
)
test_set_plot <- ggplot(comparison, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(
    title = "Actual vs. Predicted Clicks for Test Set",
    x = "Date",
    y = "Clicks",
    color = "Legend"
  ) +
  theme_minimal()
print(test_set_plot)

# Residuals
residuals <- actual - predicted

# Residuals Over Time
residual_time_plot <- ggplot(data.frame(Date = test_data$ds, Residuals = residuals), aes(x = Date, y = Residuals)) +
  geom_line(color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Residuals Over Time",
    x = "Date",
    y = "Residuals"
  ) +
  theme_minimal()
print(residual_time_plot)

# Residual Distribution
residual_dist_plot <- ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(binwidth = 50, fill = "blue", color = "black") +
  labs(
    title = "Residual Distribution",
    x = "Residuals",
    y = "Frequency"
  ) +
  theme_minimal()
print(residual_dist_plot)

# Step 7: Plot Results and Components
# --------------------------------------
# Use Prophet's native plot for forecast visualization
forecast_plot <- plot(model, forecast) +
  labs(
    title = "Forecasted Website Traffic with Confidence Intervals",
    x = "Date",
    y = "Clicks"
  ) +
  theme_minimal()
print(forecast_plot)

# Plot forecast components
components_plot <- prophet_plot_components(model, forecast)
print(components_plot)

# Step 8: Save Results
# --------------------------------------
forecast_summary <- forecast %>%
  select(ds, yhat, yhat_lower, yhat_upper)
write.csv(forecast_summary, "web_traffic_forecast_results.csv", row.names = FALSE)

message("Script completed successfully.")


