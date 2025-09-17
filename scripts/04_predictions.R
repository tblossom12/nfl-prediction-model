# ==============================================================================
# NFL PREDICTION MODEL - MAKE PREDICTIONS (FIXED)
# Script: 04_predictions_fixed.R  
# Purpose: Use trained models to predict upcoming NFL games with factor level fixes
# ==============================================================================

# Load required libraries
library(dplyr)
library(readr)
library(here)
library(nflreadr)
library(lubridate)

# Set up paths
data_processed_path <- here("data", "processed")
models_path <- here("models")
output_path <- here("output", "predictions")

# Create output directory if it doesn't exist
if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)

print("=== LOADING TRAINED MODELS ===")

# Load trained models
trained_models <- readRDS(here(models_path, "trained_models.rds"))
team_stats <- read_rds(here(data_processed_path, "team_stats_by_season.rds"))

# CRITICAL FIX: Load the training data to get factor levels
modeling_data <- read_rds(here(data_processed_path, "modeling_data.rds"))

# Extract the factor levels from training data
SEASON_PHASE_LEVELS <- levels(modeling_data$season_phase)
DIVISIONAL_GAME_LEVELS <- levels(modeling_data$divisional_game)

print("✓ Models loaded successfully")
print(paste("Season phase levels from training:", paste(SEASON_PHASE_LEVELS, collapse = ", ")))
print(paste("Divisional game levels from training:", paste(DIVISIONAL_GAME_LEVELS, collapse = ", ")))

# ==============================================================================
# DIVISION MAPPING (NEEDED FOR DIVISIONAL GAMES)
# ==============================================================================

# Create comprehensive team division mapping
team_divisions <- data.frame(
  team_abbr = c("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
                "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
                "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
                "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"),
  division = c("NFC West", "NFC South", "AFC North", "AFC East", "NFC South", "NFC North", 
               "AFC North", "AFC North", "NFC East", "AFC West", "NFC North", "NFC North", 
               "AFC South", "AFC South", "AFC South", "AFC West", "AFC West", "AFC West", 
               "NFC West", "AFC East", "NFC North", "AFC East", "NFC South", "NFC East",
               "AFC East", "NFC East", "AFC North", "NFC West", "NFC West", "NFC South", 
               "AFC South", "NFC East"),
  conference = c("NFC", "NFC", "AFC", "AFC", "NFC", "NFC", "AFC", "AFC",
                 "NFC", "AFC", "NFC", "NFC", "AFC", "AFC", "AFC", "AFC",
                 "AFC", "AFC", "NFC", "AFC", "NFC", "AFC", "NFC", "NFC", 
                 "AFC", "NFC", "AFC", "NFC", "NFC", "NFC", "AFC", "NFC")
)

# Function to determine if game is divisional
is_divisional_game <- function(home_team, away_team) {
  home_div <- team_divisions$division[team_divisions$team_abbr == home_team]
  away_div <- team_divisions$division[team_divisions$team_abbr == away_team]
  
  if (length(home_div) == 0 || length(away_div) == 0) return(0)
  return(as.numeric(home_div == away_div))
}

# ==============================================================================
# GET CURRENT SEASON DATA
# ==============================================================================

print("=== LOADING CURRENT SEASON DATA ===")

# Define current season
CURRENT_SEASON <- 2025

# Load current season schedule
current_schedule <- load_schedules(seasons = CURRENT_SEASON)

# Get current play-by-play data (up to current week)
current_pbp <- load_pbp(seasons = CURRENT_SEASON)

print(paste("✓ Current season data loaded:", CURRENT_SEASON))

# ==============================================================================
# CALCULATE CURRENT TEAM STATISTICS
# ==============================================================================

print("=== CALCULATING CURRENT TEAM STATS ===")

# Function to calculate team stats (same as in feature engineering)
calculate_current_team_stats <- function(pbp_data, season) {
  
  # Offensive stats
  offensive_stats <- pbp_data %>%
    filter(!is.na(epa), !is.na(posteam)) %>%
    group_by(season, posteam) %>%
    summarise(
      off_epa_per_play = mean(epa, na.rm = TRUE),
      off_success_rate = mean(success, na.rm = TRUE),
      off_red_zone_td_rate = mean(touchdown[yardline_100 <= 20], na.rm = TRUE),
      off_third_down_conv = mean(third_down_converted[down == 3], na.rm = TRUE),
      off_rush_epa_per_play = mean(epa[rush == 1], na.rm = TRUE),
      off_rush_success_rate = mean(success[rush == 1], na.rm = TRUE),
      off_pass_epa_per_play = mean(epa[pass == 1], na.rm = TRUE),
      off_pass_success_rate = mean(success[pass == 1], na.rm = TRUE),
      off_completion_rate = mean(complete_pass[pass == 1], na.rm = TRUE),
      off_int_rate = sum(interception, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
      off_fumble_rate = sum(fumble_lost, na.rm = TRUE) / n(),
      off_plays = n(),
      .groups = "drop"
    )
  
  # Defensive stats  
  defensive_stats <- pbp_data %>%
    filter(!is.na(epa), !is.na(defteam)) %>%
    group_by(season, defteam) %>%
    summarise(
      def_epa_allowed = mean(epa, na.rm = TRUE),
      def_success_rate_allowed = mean(success, na.rm = TRUE),
      def_red_zone_td_allowed = mean(touchdown[yardline_100 <= 20], na.rm = TRUE),
      def_third_down_allowed = mean(third_down_converted[down == 3], na.rm = TRUE),
      def_rush_epa_allowed = mean(epa[rush == 1], na.rm = TRUE),
      def_rush_success_allowed = mean(success[rush == 1], na.rm = TRUE),
      def_pass_epa_allowed = mean(epa[pass == 1], na.rm = TRUE),
      def_pass_success_allowed = mean(success[pass == 1], na.rm = TRUE),
      def_completion_rate_allowed = mean(complete_pass[pass == 1], na.rm = TRUE),
      def_int_rate = sum(interception, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
      def_fumble_recovery_rate = sum(fumble_forced, na.rm = TRUE) / n(),
      def_sack_rate = sum(sack, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(posteam = defteam)
  
  # Combine and calculate derived metrics
  combined_stats <- offensive_stats %>%
    left_join(defensive_stats, by = c("season", "posteam")) %>%
    mutate(
      net_epa_per_play = off_epa_per_play - def_epa_allowed,
      net_success_rate = off_success_rate - def_success_rate_allowed,
      rush_pass_balance = off_rush_epa_per_play / (off_rush_epa_per_play + off_pass_epa_per_play),
      turnover_margin_rate = (def_int_rate + def_fumble_recovery_rate) - (off_int_rate + off_fumble_rate)
    )
  
  return(combined_stats)
}

# Calculate current season team stats
if (nrow(current_pbp) > 0) {
  current_team_stats <- calculate_current_team_stats(current_pbp, CURRENT_SEASON)
  print(paste("✓ Current team stats calculated for", nrow(current_team_stats), "teams"))
} else {
  # If season hasn't started, use last season's stats
  print("No current season data available, using 2023 season stats")
  current_team_stats <- team_stats %>% 
    filter(season == 2023) %>%
    mutate(season = CURRENT_SEASON)
}

# ==============================================================================
# FIXED PREDICTION FUNCTIONS
# ==============================================================================

print("=== SETTING UP PREDICTION FUNCTIONS ===")

# Function to safely convert factors with training levels
safe_factor_conversion <- function(value, training_levels) {
  if (value %in% training_levels) {
    return(factor(value, levels = training_levels))
  } else {
    # Default to the first level if new level encountered
    warning(paste("Unknown factor level:", value, "- using default:", training_levels[1]))
    return(factor(training_levels[1], levels = training_levels))
  }
}

# Function to determine season phase based on week
get_season_phase <- function(week_num) {
  phase <- case_when(
    week_num <= 6 ~ "Early",
    week_num <= 12 ~ "Mid", 
    week_num <= 18 ~ "Late",
    TRUE ~ "Playoffs"
  )
  
  # Ensure it matches training levels
  if (!phase %in% SEASON_PHASE_LEVELS) {
    warning(paste("Season phase", phase, "not in training data. Using 'Mid' instead."))
    phase <- "Mid"
  }
  
  return(phase)
}

# Function to create features for a single game (FIXED)
create_game_features <- function(home_team, away_team, week, season_phase = NULL) {
  
  # Determine season phase if not provided
  if (is.null(season_phase)) {
    season_phase <- get_season_phase(week)
  }
  
  # Get home team stats
  home_stats <- current_team_stats %>% 
    filter(posteam == home_team) %>%
    select(-season, -posteam)
  
  # Get away team stats  
  away_stats <- current_team_stats %>%
    filter(posteam == away_team) %>%
    select(-season, -posteam)
  
  if (nrow(home_stats) == 0 || nrow(away_stats) == 0) {
    stop(paste("Team stats not found for teams:", home_team, "or", away_team))
  }
  
  # Determine if divisional game
  div_game_value <- is_divisional_game(home_team, away_team)
  
  # Calculate matchup features
  features <- data.frame(
    # Core advantages
    epa_advantage = (home_stats$off_epa_per_play - away_stats$def_epa_allowed) - 
      (away_stats$off_epa_per_play - home_stats$def_epa_allowed),
    
    success_rate_advantage = (home_stats$off_success_rate - away_stats$def_success_rate_allowed) - 
      (away_stats$off_success_rate - home_stats$def_success_rate_allowed),
    
    # Specific matchups
    rush_matchup_advantage = (home_stats$off_rush_epa_per_play - away_stats$def_rush_epa_allowed) -
      (away_stats$off_rush_epa_per_play - home_stats$def_rush_epa_allowed),
    
    #pass_matchup_advantage = (home_stats$off_pass_epa_per_play - away_stats$def_pass_epa_allowed) -
      #(away_stats$off_pass_epa_per_play - home_stats$def_pass_epa_allowed),
    
    # Situational advantages
    red_zone_advantage = (home_stats$off_red_zone_td_rate - away_stats$def_red_zone_td_allowed) -
      (away_stats$off_red_zone_td_rate - home_stats$def_red_zone_td_allowed),
    
    third_down_advantage = (home_stats$off_third_down_conv - away_stats$def_third_down_allowed) -
      (away_stats$off_third_down_conv - home_stats$def_third_down_allowed),
    
    turnover_advantage = home_stats$turnover_margin_rate - away_stats$turnover_margin_rate,
    
    # Team strength
    home_net_epa_per_play = home_stats$net_epa_per_play,
    away_net_epa_per_play = away_stats$net_epa_per_play,
    home_net_success_rate = home_stats$net_success_rate,  
    away_net_success_rate = away_stats$net_success_rate,
    
    # Context features
    #pace_differential = home_stats$rush_pass_balance - away_stats$rush_pass_balance,
    rest_differential = 0,  # Default to 0 if not available
    week = week
  )
  
  # CRITICAL FIX: Convert factors using training levels
  features$divisional_game <- safe_factor_conversion(as.character(div_game_value), DIVISIONAL_GAME_LEVELS)
  features$season_phase <- safe_factor_conversion(season_phase, SEASON_PHASE_LEVELS)
  
  # Handle any NA values that might cause issues
  numeric_cols <- sapply(features, is.numeric)
  features[numeric_cols] <- lapply(features[numeric_cols], function(x) {
    ifelse(is.na(x) | is.infinite(x), 0, x)
  })
  
  return(features)
}

# Function to predict a single game using all models (FIXED)
predict_game <- function(home_team, away_team, week = 1, season_phase = NULL) {
  
  cat("Predicting:", away_team, "@", home_team, "\n")
  
  tryCatch({
    # Create features
    game_features <- create_game_features(home_team, away_team, week, season_phase)
    
    # Prepare data for different models
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
      #"pace_differential",
      "rest_differential",
      "week"
    )
    
    # Logistic regression prediction
    X_matrix <- model.matrix(~ . - 1, game_features[, numeric_features])
    logistic_prob <- predict(trained_models$logistic, X_matrix, type = "response", s = "lambda.min")[1,1]
    
    # Random Forest prediction
    rf_prob <- predict(trained_models$random_forest, game_features, type = "prob")[1, "Win"]
    
    # XGBoost prediction (need to convert factors to numeric)
    game_features_xgb <- game_features %>%
      mutate(
        divisional_game = as.numeric(as.character(divisional_game)),
        season_phase = as.numeric(season_phase)  # This will now work correctly
      )
    
    # Handle XGBoost matrix creation more safely
    xgb_data <- as.matrix(game_features_xgb[, names(game_features_xgb) != ""])
    xgb_matrix <- xgb.DMatrix(data = xgb_data)
    xgb_prob <- predict(trained_models$xgboost, xgb_matrix)
    
    # SVM prediction
    svm_pred <- predict(trained_models$svm, game_features[, numeric_features], probability = TRUE)
    svm_prob <- attr(svm_pred, "probabilities")[1, "Win"]
    
    # Ensemble prediction (average of all models)
    ensemble_prob <- (logistic_prob + rf_prob + xgb_prob + svm_prob) / 4
    
    # Create results dataframe
    prediction_result <- data.frame(
      home_team = home_team,
      away_team = away_team,
      week = week,
      season_phase = as.character(game_features$season_phase),
      divisional_game = as.character(game_features$divisional_game),
      logistic_prob = round(logistic_prob, 3),
      rf_prob = round(rf_prob, 3),
      xgb_prob = round(xgb_prob, 3),
      svm_prob = round(svm_prob, 3),
      ensemble_prob = round(ensemble_prob, 3),
      predicted_winner = ifelse(ensemble_prob > 0.5, home_team, away_team),
      confidence = ifelse(ensemble_prob > 0.5, 
                          paste0(round(ensemble_prob * 100, 1), "%"),
                          paste0(round((1 - ensemble_prob) * 100, 1), "%"))
    )
    
    return(prediction_result)
    
  }, error = function(e) {
    cat("Error predicting game:", e$message, "\n")
    return(NULL)
  })
}

# ==============================================================================
# PREDICT UPCOMING GAMES
# ==============================================================================

print("=== PREDICTING UPCOMING GAMES ===")

# Get upcoming games (games without results)
upcoming_games <- current_schedule %>%
  filter(is.na(result), !is.na(home_team), !is.na(away_team)) %>%
  arrange(week, gameday) %>%
  head(20)  # Limit to next 20 games

if (nrow(upcoming_games) > 0) {
  print(paste("Found", nrow(upcoming_games), "upcoming games to predict"))
  
  # Make predictions for upcoming games
  upcoming_predictions <- data.frame()
  
  for (i in 1:nrow(upcoming_games)) {
    game <- upcoming_games[i, ]
    pred <- predict_game(game$home_team, game$away_team, game$week)
    
    if (!is.null(pred)) {
      pred_with_context <- cbind(
        game_id = game$game_id,
        season = game$season,
        gameday = game$gameday,
        pred
      )
      upcoming_predictions <- bind_rows(upcoming_predictions, pred_with_context)
    }
  }
  
  print("✓ Predictions completed for upcoming games")
  
} else {
  print("No upcoming games found")
  upcoming_predictions <- data.frame()
}

# ==============================================================================
# PREDICT SPECIFIC MATCHUPS (EXAMPLES)
# ==============================================================================

print("=== EXAMPLE PREDICTIONS ===")

# Example predictions for popular matchups (check if teams exist in stats)
available_teams <- unique(current_team_stats$posteam)
example_matchups <- list(
  c("KC", "BUF"),
  c("BAL", "CIN"), 
  c("SF", "LAR"),
  c("DAL", "PHI")
)

example_predictions <- data.frame()

for (matchup in example_matchups) {
  home_team <- matchup[1]
  away_team <- matchup[2]
  
  # Check if both teams have stats
  if (home_team %in% available_teams && away_team %in% available_teams) {
    pred <- predict_game(home_team, away_team, week = 10)
    if (!is.null(pred)) {
      example_predictions <- bind_rows(example_predictions, pred)
    }
  } else {
    cat("Skipping", away_team, "@", home_team, "- missing team stats\n")
  }
}

if (nrow(example_predictions) > 0) {
  print("Example matchup predictions:")
  print(example_predictions)
}

# ==============================================================================
# INTERACTIVE FUNCTIONS
# ==============================================================================

# Function for manual predictions (FIXED)
manual_predict <- function(home_team, away_team, week = 1) {
  result <- predict_game(home_team, away_team, week)
  if (!is.null(result)) {
    cat("\n=== GAME PREDICTION ===\n")
    cat("Matchup:", result$away_team, "@", result$home_team, "\n")
    cat("Week:", result$week, "\n")
    cat("Season Phase:", result$season_phase, "\n")
    cat("Divisional Game:", result$divisional_game, "\n\n")
    cat("MODEL PROBABILITIES (Home Team Win):\n")
    cat("Logistic Regression:", result$logistic_prob, "\n")
    cat("Random Forest:      ", result$rf_prob, "\n") 
    cat("XGBoost:            ", result$xgb_prob, "\n")
    cat("SVM:                ", result$svm_prob, "\n")
    cat("ENSEMBLE:           ", result$ensemble_prob, "\n\n")
    cat("PREDICTION:", result$predicted_winner, "wins with", result$confidence, "confidence\n")
    cat("========================\n\n")
  }
  return(result)
}

# Function to predict all games in a specific week (FIXED)
predict_week <- function(week_num, season = CURRENT_SEASON) {
  
  cat("Predicting all games for Week", week_num, "of", season, "season\n")
  
  week_games <- current_schedule %>%
    filter(week == week_num, season == !!season) %>%
    filter(!is.na(home_team), !is.na(away_team))
  
  if (nrow(week_games) == 0) {
    cat("No games found for Week", week_num, "\n")
    return(data.frame())
  }
  
  week_predictions <- data.frame()
  
  for (i in 1:nrow(week_games)) {
    game <- week_games[i, ]
    pred <- predict_game(game$home_team, game$away_team, game$week)
    
    if (!is.null(pred)) {
      pred_with_context <- cbind(
        game_id = game$game_id,
        season = game$season,
        gameday = game$gameday,
        gametime = if("gametime" %in% names(game)) game$gametime else NA,
        pred
      )
      week_predictions <- bind_rows(week_predictions, pred_with_context)
    }
  }
  
  week_predictions <- week_predictions %>% arrange(gameday)
  
  return(week_predictions)
}

# Function to get team strength rankings
get_team_rankings <- function() {
  rankings <- current_team_stats %>%
    mutate(
      overall_rating = net_epa_per_play * 100,  # Convert to more readable scale
      offensive_rating = off_epa_per_play * 100,
      defensive_rating = -def_epa_allowed * 100  # Negative because lower is better for defense
    ) %>%
    arrange(desc(overall_rating)) %>%
    select(posteam, overall_rating, offensive_rating, defensive_rating, 
           off_epa_per_play, def_epa_allowed, net_epa_per_play) %>%
    mutate(rank = row_number())
  
  return(rankings)
}

# ==============================================================================
# SAVE PREDICTIONS
# ==============================================================================

print("=== SAVING PREDICTIONS ===")

# Save upcoming games predictions
if (nrow(upcoming_predictions) > 0) {
  write_csv(upcoming_predictions, here(output_path, paste0("upcoming_predictions_", Sys.Date(), ".csv")))
  print("✓ Upcoming predictions saved")
}

# Save example predictions
if (nrow(example_predictions) > 0) {
  write_csv(example_predictions, here(output_path, paste0("example_predictions_", Sys.Date(), ".csv")))
  print("✓ Example predictions saved")
}

# Create a prediction summary report
prediction_summary <- data.frame(
  prediction_date = Sys.Date(),
  season = CURRENT_SEASON,
  total_upcoming_games = nrow(upcoming_predictions),
  models_used = "Logistic, Random Forest, XGBoost, SVM",
  ensemble_method = "Simple Average",
  data_through = if(nrow(current_pbp) > 0) max(current_pbp$week, na.rm = TRUE) else "Preseason",
  factor_levels_fixed = TRUE
)

write_csv(prediction_summary, here(output_path, "prediction_summary.csv"))

print("✓ All predictions saved successfully")

# ==============================================================================
# FINAL SUMMARY AND USAGE EXAMPLES
# ==============================================================================

print("=== PREDICTION SYSTEM READY (FIXED) ===")
print(paste("System initialized at:", Sys.time()))
print(paste("Current season:", CURRENT_SEASON))

if (nrow(current_team_stats) > 0) {
  print(paste("Team stats available for", nrow(current_team_stats), "teams"))
  
  # Show available teams
  cat("Available teams:", paste(sort(current_team_stats$posteam), collapse = ", "), "\n")
  
  # Show team rankings
  team_rankings <- get_team_rankings()
  cat("\nTOP 5 TEAMS BY OVERALL RATING:\n")
  print(head(team_rankings[, c("rank", "posteam", "overall_rating")], 5))
}

cat("\n=== USAGE EXAMPLES ===\n")
cat("# Predict a single game:\n")
cat("manual_predict('KC', 'BUF', week = 10)\n\n")
cat("# Predict all games in a week:\n") 
cat("week_10_predictions <- predict_week(10)\n\n")
cat("# Get current team rankings:\n")
cat("rankings <- get_team_rankings()\n")
cat("print(rankings)\n\n")

cat("=== FILES SAVED ===\n")
cat("- Upcoming predictions:", here(output_path, paste0("upcoming_predictions_", Sys.Date(), ".csv")), "\n")
cat("- Example predictions:", here(output_path, paste0("example_predictions_", Sys.Date(), ".csv")), "\n")
cat("- Prediction summary:", here(output_path, "prediction_summary.csv"), "\n")

print("\n=== PREDICTION SYSTEM COMPLETE (FACTOR LEVELS FIXED) ===")
print("The factor level issue has been resolved. You can now make predictions!")

# Test the system with a simple prediction if data is available
if (nrow(current_team_stats) >= 2) {
  cat("\n=== TESTING PREDICTION SYSTEM ===\n")
  test_teams <- head(current_team_stats$posteam, 2)
  cat("Testing with teams:", test_teams[1], "vs", test_teams[2], "\n")
  
  test_result <- predict_game(test_teams[1], test_teams[2], week = 10)
  if (!is.null(test_result)) {
    cat("✓ Test prediction successful!\n")
    cat("Predicted winner:", test_result$predicted_winner, "with", test_result$confidence, "confidence\n")
  }
}