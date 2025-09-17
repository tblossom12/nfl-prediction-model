# ==============================================================================
# NFL PREDICTION MODEL - MODEL TRAINING
# Script: 03_model_training.R
# Purpose: Train and evaluate multiple ML models for NFL game prediction
# ==============================================================================

# Load required libraries
library(dplyr)
library(readr)
library(here)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(e1071)
library(yardstick)
library(broom)

# Set up paths
data_processed_path <- here("data", "processed")
models_path <- here("models")

# Create models directory if it doesn't exist
if (!dir.exists(models_path)) dir.create(models_path, recursive = TRUE)

print("=== LOADING PROCESSED DATA ===")

# Load modeling data
modeling_data <- read_rds(here(data_processed_path, "modeling_data.rds"))

print(paste("✓ Loaded", nrow(modeling_data), "games for modeling"))

# ==============================================================================
# DATA PREPARATION FOR MODELING
# ==============================================================================

print("=== PREPARING DATA FOR MODELING ===")

# Set seed for reproducibility
set.seed(42)

# Define features for modeling (excluding categorical for some models)
numeric_features <- c(
  "epa_advantage",
  "success_rate_advantage",
  "rush_matchup_advantage",
  #"pass_matchup_advantage",
  "red_zone_advantage",
  "third_down_advantage",
  "turnover_advantage",
  "home_net_epa_per_play",
  "away_net_epa_per_play",
  "home_net_success_rate",
  "away_net_success_rate",
  "pressure_differential",
  #"pace_differential",
  "rest_differential",
  "week"
)

all_features <- c(numeric_features, "divisional_game", "season_phase")

# Create training/validation/test splits by season
train_data <- modeling_data %>% filter(season <= 2022)  # Changed from 2021 (2020-2022 for training)
val_data <- modeling_data %>% filter(season == 2023)    # Changed from 2022 (2023 for validation)  
test_data <- modeling_data %>% filter(season == 2024)   # Changed from 2023 (2024 for final testing)

print(paste("Training set:", nrow(train_data), "games"))
print(paste("Validation set:", nrow(val_data), "games"))
print(paste("Test set:", nrow(test_data), "games"))

# Prepare feature matrices
X_train <- train_data[, all_features]
y_train <- train_data$home_win
X_val <- val_data[, all_features]
y_val <- val_data$home_win
X_test <- test_data[, all_features]
y_test <- test_data$home_win

# For models requiring numeric data only
X_train_numeric <- train_data[, numeric_features]
X_val_numeric <- val_data[, numeric_features]
X_test_numeric <- test_data[, numeric_features]

# ==============================================================================
# MODEL TRAINING CONFIGURATION
# ==============================================================================

print("=== CONFIGURING MODEL TRAINING ===")

# Set up cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Initialize results storage
model_results <- list()
model_objects <- list()

# ==============================================================================
# MODEL 1: LOGISTIC REGRESSION WITH REGULARIZATION
# ==============================================================================

print("=== TRAINING LOGISTIC REGRESSION ===")

# Prepare data for glmnet (requires matrix format)
X_train_matrix <- model.matrix(~ . - 1, X_train_numeric)
X_val_matrix <- model.matrix(~ . - 1, X_val_numeric)
X_test_matrix <- model.matrix(~ . - 1, X_test_numeric)

# Train elastic net logistic regression
logistic_model <- cv.glmnet(
  x = X_train_matrix,
  y = y_train,
  family = "binomial",
  alpha = 0.5,  # Elastic net (0.5 between ridge and lasso)
  nfolds = 5,
  type.measure = "class"
)

# Make predictions
pred_logistic_val <- predict(logistic_model, X_val_matrix, type = "response", s = "lambda.min")[,1]
pred_logistic_test <- predict(logistic_model, X_test_matrix, type = "response", s = "lambda.min")[,1]

# Convert to class predictions
pred_logistic_val_class <- factor(ifelse(pred_logistic_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
pred_logistic_test_class <- factor(ifelse(pred_logistic_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

# Store results
model_objects$logistic <- logistic_model
model_results$logistic <- list(
  val_prob = pred_logistic_val,
  val_class = pred_logistic_val_class,
  test_prob = pred_logistic_test,
  test_class = pred_logistic_test_class
)

print("✓ Logistic regression completed")

# ==============================================================================
# MODEL 2: RANDOM FOREST
# ==============================================================================

print("=== TRAINING RANDOM FOREST ===")

# Train random forest
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = sqrt(length(all_features)),
  importance = TRUE,
  do.trace = 100  # Progress updates every 100 trees
)

# Make predictions
pred_rf_val <- predict(rf_model, X_val, type = "prob")[, "Win"]
pred_rf_val_class <- predict(rf_model, X_val, type = "class")
pred_rf_test <- predict(rf_model, X_test, type = "prob")[, "Win"]
pred_rf_test_class <- predict(rf_model, X_test, type = "class")

# Store results
model_objects$random_forest <- rf_model
model_results$random_forest <- list(
  val_prob = pred_rf_val,
  val_class = pred_rf_val_class,
  test_prob = pred_rf_test,
  test_class = pred_rf_test_class
)

print("✓ Random Forest completed")

# ==============================================================================
# MODEL 3: XGBOOST
# ==============================================================================

print("=== TRAINING XGBOOST ===")

# Prepare data for XGBoost
# Convert factors to numeric for XGBoost
X_train_xgb <- X_train %>%
  mutate(
    divisional_game = as.numeric(as.character(divisional_game)),
    season_phase = as.numeric(as.factor(season_phase))
  )
X_val_xgb <- X_val %>%
  mutate(
    divisional_game = as.numeric(as.character(divisional_game)),
    season_phase = as.numeric(as.factor(season_phase))
  )
X_test_xgb <- X_test %>%
  mutate(
    divisional_game = as.numeric(as.character(divisional_game)),
    season_phase = as.numeric(as.factor(season_phase))
  )

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = as.matrix(X_train_xgb), label = as.numeric(y_train) - 1)
dval <- xgb.DMatrix(data = as.matrix(X_val_xgb), label = as.numeric(y_val) - 1)
dtest <- xgb.DMatrix(data = as.matrix(X_test_xgb), label = as.numeric(y_test) - 1)

# XGBoost parameters
xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  seed = 42
)

# Train with early stopping
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, val = dval),
  early_stopping_rounds = 20,
  verbose = 1
)

# Make predictions
pred_xgb_val <- predict(xgb_model, dval)
pred_xgb_val_class <- factor(ifelse(pred_xgb_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
pred_xgb_test <- predict(xgb_model, dtest)
pred_xgb_test_class <- factor(ifelse(pred_xgb_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

# Store results
model_objects$xgboost <- xgb_model
model_results$xgboost <- list(
  val_prob = pred_xgb_val,
  val_class = pred_xgb_val_class,
  test_prob = pred_xgb_test,
  test_class = pred_xgb_test_class
)

print("✓ XGBoost completed")

# ==============================================================================
# MODEL 4: SUPPORT VECTOR MACHINE
# ==============================================================================

print("=== TRAINING SUPPORT VECTOR MACHINE ===")

# Train SVM with probability estimates
svm_model <- svm(
  x = X_train_numeric,
  y = y_train,
  type = "C-classification",
  kernel = "radial",
  probability = TRUE,
  cost = 1,
  gamma = 1/ncol(X_train_numeric)
)

# Make predictions
pred_svm_val <- predict(svm_model, X_val_numeric, probability = TRUE)
pred_svm_val_prob <- attr(pred_svm_val, "probabilities")[, "Win"]
pred_svm_test <- predict(svm_model, X_test_numeric, probability = TRUE)
pred_svm_test_prob <- attr(pred_svm_test, "probabilities")[, "Win"]

# Store results
model_objects$svm <- svm_model
model_results$svm <- list(
  val_prob = pred_svm_val_prob,
  val_class = pred_svm_val,
  test_prob = pred_svm_test_prob,
  test_class = pred_svm_test
)

print("✓ SVM completed")

# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

print("=== EVALUATING MODELS ===")

# Function to calculate comprehensive metrics
evaluate_model <- function(actual, predicted_prob, predicted_class, model_name, type) {
  
  # Basic accuracy
  accuracy <- mean(actual == predicted_class)
  
  # AUC
  roc_obj <- yardstick::roc_auc_vec(actual, predicted_prob)
  
  # Log loss
  log_loss <- yardstick::mn_log_loss_vec(actual, predicted_prob)
  
  # Confusion matrix metrics
  cm <- table(Predicted = predicted_class, Actual = actual)
  precision <- cm[2,2] / sum(cm[2,])
  recall <- cm[2,2] / sum(cm[,2])
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  return(data.frame(
    Model = model_name,
    Type = type,
    Accuracy = round(accuracy, 4),
    AUC = 1-round(roc_obj, 4),
    LogLoss = round(log_loss, 4),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1_Score = round(f1, 4)
  ))
}

# Evaluate all models on validation set
val_results <- bind_rows(
  evaluate_model(y_val, model_results$logistic$val_prob, model_results$logistic$val_class, "Logistic", "val"),
  evaluate_model(y_val, model_results$random_forest$val_prob, model_results$random_forest$val_class, "Random Forest", "val"),
  evaluate_model(y_val, model_results$xgboost$val_prob, model_results$xgboost$val_class, "XGBoost", "val"),
  evaluate_model(y_val, model_results$svm$val_prob, model_results$svm$val_class, "SVM", "val")
)

# Evaluate all models on test set
test_results <- bind_rows(
  evaluate_model(y_test, model_results$logistic$test_prob, model_results$logistic$test_class, "Logistic", "test"),
  evaluate_model(y_test, model_results$random_forest$test_prob, model_results$random_forest$test_class, "Random Forest", "test"),
  evaluate_model(y_test, model_results$xgboost$test_prob, model_results$xgboost$test_class, "XGBoost", "test"),
  evaluate_model(y_test, model_results$svm$test_prob, model_results$svm$test_class, "SVM", "test")
)

print("VALIDATION SET PERFORMANCE:")
print(val_results)
print("\nTEST SET PERFORMANCE:")
print(test_results)

# ==============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

print("=== ANALYZING FEATURE IMPORTANCE ===")

# Random Forest feature importance
rf_importance <- importance(model_objects$random_forest) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(Model = "Random Forest", Importance = MeanDecreaseGini) %>%
  select(Model, Feature, Importance)

# XGBoost feature importance
xgb_importance <- xgb.importance(model = model_objects$xgboost) %>%
  mutate(Model = "XGBoost", Importance = Gain) %>%
  select(Model, Feature, Importance)

# Combine importance scores
feature_importance <- bind_rows(rf_importance[1:10,], xgb_importance[1:10,])

print("TOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance)


# ==============================================================================
# AUTOMATED FEATURE SELECTION AND OPTIMIZATION
# ==============================================================================

print("=== FEATURE OPTIMIZATION PROCESS ===")

# 1. BASELINE MODEL PERFORMANCE
baseline_accuracy <- ensemble_test_metrics$Accuracy

# 2. IDENTIFY LOW IMPACT FEATURES
# Combine Random Forest and XGBoost importance scores
combined_importance <- feature_importance %>%
  group_by(Feature) %>%
  summarise(avg_importance = mean(Importance, na.rm = TRUE)) %>%
  arrange(desc(avg_importance))

print("Feature importance ranking:")
print(combined_importance)

# Set thresholds for feature categories
very_high_threshold <- 0.15
high_threshold <- 0.10
medium_threshold <- 0.05

# Categorize features by importance
feature_categories <- combined_importance %>%
  mutate(
    impact_level = case_when(
      avg_importance >= very_high_threshold ~ "Very High",
      avg_importance >= high_threshold ~ "High", 
      avg_importance >= medium_threshold ~ "Medium",
      TRUE ~ "Low"
    )
  )

print("Features by impact level:")
print(feature_categories %>% count(impact_level))

# 3. TEST FEATURE SUBSETS
test_feature_combinations <- list(
  "Top_5" = feature_categories %>% slice_head(n = 5) %>% pull(Feature),
  "Medium_Plus" = feature_categories %>% filter(impact_level %in% c("Very High", "High", "Medium")) %>% pull(Feature),
  "High_Plus" = feature_categories %>% filter(impact_level %in% c("Very High", "High")) %>% pull(Feature),
  "Very_High_Only" = feature_categories %>% filter(impact_level == "Very High") %>% pull(Feature)
)

# Function to test a feature subset
test_feature_subset <- function(features_to_use, subset_name) {
  cat("Testing feature subset:", subset_name, "(", length(features_to_use), "features )\n")
  
  # Prepare data with selected features only
  X_train_subset <- X_train[, features_to_use, drop = FALSE]
  X_test_subset <- X_test[, features_to_use, drop = FALSE]
  
  # Quick Random Forest model (fastest to train)
  rf_subset <- randomForest(
    x = X_train_subset,
    y = y_train,
    ntree = 100,  # Reduced for speed
    mtry = max(1, sqrt(length(features_to_use)))
  )
  
  # Test predictions
  pred_subset <- predict(rf_subset, X_test_subset, type = "prob")[, "Win"]
  pred_subset_class <- predict(rf_subset, X_test_subset, type = "class")
  
  # Calculate accuracy
  accuracy_subset <- mean(y_test == pred_subset_class)
  
  return(data.frame(
    subset_name = subset_name,
    num_features = length(features_to_use),
    accuracy = accuracy_subset,
    improvement = accuracy_subset - baseline_accuracy
  ))
}

# Test all feature combinations
subset_results <- do.call(rbind, lapply(names(test_feature_combinations), function(name) {
  test_feature_subset(test_feature_combinations[[name]], name)
}))

print("Feature subset performance:")
print(subset_results %>% arrange(desc(accuracy)))

# 4. RECOMMEND OPTIMAL FEATURE SET
best_subset <- subset_results %>% slice_max(accuracy, n = 1)
optimal_features <- test_feature_combinations[[best_subset$subset_name]]

cat("\n=== FEATURE OPTIMIZATION RESULTS ===\n")
cat("Best feature subset:", best_subset$subset_name, "\n")
cat("Number of features:", best_subset$num_features, "\n") 
cat("Accuracy improvement:", round(best_subset$improvement * 100, 2), "%\n")

cat("\nOptimal feature set:\n")
cat(paste("-", optimal_features), sep = "\n")

# Save optimization results
feature_optimization_results <- list(
  baseline_accuracy = baseline_accuracy,
  feature_categories = feature_categories,
  subset_results = subset_results,
  optimal_features = optimal_features
)

saveRDS(feature_optimization_results, here(models_path, "feature_optimization.rds"))
write_csv(feature_categories, here(models_path, "feature_importance_categories.csv"))

print("Feature optimization results saved!")
# ==============================================================================
# ENSEMBLE MODEL
# ==============================================================================

print("=== CREATING ENSEMBLE MODEL ===")

# Simple average ensemble of top 3 models
ensemble_val <- (model_results$random_forest$val_prob + 
                   model_results$xgboost$val_prob + 
                   model_results$logistic$val_prob) / 3

ensemble_test <- (model_results$random_forest$test_prob + 
                    model_results$xgboost$test_prob + 
                    model_results$logistic$test_prob) / 3

ensemble_val_class <- factor(ifelse(ensemble_val > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
ensemble_test_class <- factor(ifelse(ensemble_test > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

# Evaluate ensemble
ensemble_val_metrics <- evaluate_model(y_val, ensemble_val, ensemble_val_class, "Ensemble", "val")
ensemble_test_metrics <- evaluate_model(y_test, ensemble_test, ensemble_test_class, "Ensemble", "test")

print("ENSEMBLE MODEL PERFORMANCE:")
cat("Validation:\n")
print(ensemble_val_metrics)
cat("Test:\n")
print(ensemble_test_metrics)

# ==============================================================================
# SAVE MODELS AND RESULTS
# ==============================================================================

print("=== SAVING MODELS AND RESULTS ===")

# Save model objects
saveRDS(model_objects, here(models_path, "trained_models.rds"))

# Save predictions
predictions_summary <- data.frame(
  game_id = test_data$game_id,
  season = test_data$season,
  week = test_data$week,
  home_team = test_data$home_team,
  away_team = test_data$away_team,
  actual_result = y_test,
  logistic_prob = model_results$logistic$test_prob,
  rf_prob = model_results$random_forest$test_prob,
  xgb_prob = model_results$xgboost$test_prob,
  svm_prob = model_results$svm$test_prob,
  ensemble_prob = ensemble_test,
  ensemble_prediction = ensemble_test_class
)

write_csv(predictions_summary, here(models_path, "test_predictions.csv"))

# Save performance metrics
write_csv(bind_rows(val_results, test_results, ensemble_val_metrics, ensemble_test_metrics), 
          here(models_path, "model_performance.csv"))

feature_importance = bind_rows(rf_importance, xgb_importance)
# Save feature importance
write_csv(feature_importance, here(models_path, "feature_importance.csv"))

print("✓ All models and results saved")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("=== MODEL TRAINING SUMMARY ===")
print(paste("Training completed at:", Sys.time()))

# Find best model on test set
best_model <- test_results[which.max(test_results$AUC), ]
print(paste("Best performing model:", best_model$Model))
print(paste("Test set AUC:", best_model$AUC))
print(paste("Test set Accuracy:", best_model$Accuracy))

print("\n=== READY FOR PREDICTIONS ===")
print("Next step: Run 04_predictions.R for making new predictions")
print(paste("Models saved in:", models_path))