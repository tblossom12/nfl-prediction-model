# ==============================================================================
# NFL PREDICTION MODEL - FEATURE ENGINEERING  
# Script: 02_feature_engineering.R
# Purpose: Transform raw NFL data into model-ready features
# ==============================================================================

# Load required libraries
library(dplyr)
library(tidyr)
library(readr)
library(here)
library(lubridate)

# Set up paths
data_raw_path <- here("data", "raw")
data_processed_path <- here("data", "processed")

print("=== LOADING RAW DATA ===")

# Load raw data
games_raw <- read_rds(here(data_raw_path, "games_raw.rds"))
pbp_raw <- read_rds(here(data_raw_path, "pbp_raw.rds"))
teams <- read_rds(here(data_raw_path, "teams.rds"))

# Load injury data
team_injury_summary <- read_rds(here(data_raw_path, "team_injury_summary.rds"))
injuries_processed <- read_rds(here(data_raw_path, "injuries_processed.rds"))

print("✓ Raw data loaded successfully")

# ==============================================================================
# ENHANCED TEAM STATISTICS WITH INJURY CONTEXT
# ==============================================================================

print("=== CALCULATING ENHANCED TEAM STATISTICS ===")

# Original offensive statistics
offensive_stats <- pbp_raw %>%
  filter(!is.na(epa), !is.na(posteam)) %>%
  group_by(season, posteam) %>%
  summarise(
    # Core offensive metrics
    off_plays = n(),
    off_epa_per_play = mean(epa, na.rm = TRUE),
    off_success_rate = mean(success, na.rm = TRUE),
    off_explosive_rate = mean(epa > 0.5, na.rm = TRUE),
    
    # Situational offense
    off_red_zone_plays = sum(yardline_100 <= 20, na.rm = TRUE),
    off_red_zone_td_rate = mean(touchdown[yardline_100 <= 20], na.rm = TRUE),
    off_third_down_plays = sum(down == 3, na.rm = TRUE),
    off_third_down_conv = mean(third_down_converted[down == 3], na.rm = TRUE),
    off_fourth_down_conv = mean(fourth_down_converted[down == 4], na.rm = TRUE),
    
    # Rushing offense
    off_rush_plays = sum(rush == 1, na.rm = TRUE),
    off_rush_epa_per_play = mean(epa[rush == 1], na.rm = TRUE),
    off_rush_success_rate = mean(success[rush == 1], na.rm = TRUE),
    off_rush_yards_per_play = mean(yards_gained[rush == 1], na.rm = TRUE),
    
    # Passing offense
    off_pass_plays = sum(pass == 1, na.rm = TRUE),
    off_pass_epa_per_play = mean(epa[pass == 1], na.rm = TRUE),
    off_pass_success_rate = mean(success[pass == 1], na.rm = TRUE),
    off_completion_rate = mean(complete_pass[pass == 1], na.rm = TRUE),
    off_pass_yards_per_play = mean(yards_gained[pass == 1], na.rm = TRUE),
    off_air_yards_per_att = mean(air_yards[pass == 1], na.rm = TRUE),
    off_yac_per_comp = mean(yards_after_catch[complete_pass == 1], na.rm = TRUE),
    
    # NEW: Performance variability metrics
    off_epa_variance = var(epa, na.rm = TRUE),
    off_epa_consistency = 1 / (1 + var(epa, na.rm = TRUE)),  # Consistency score
    
    # NEW: Pressure and protection metrics
    off_sack_rate_allowed = sum(sack, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    off_pressure_rate_allowed = sum(!is.na(qb_hit) & qb_hit == 1, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    
    # Turnover metrics
    off_int_rate = sum(interception, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    off_fumble_rate = sum(fumble_lost, na.rm = TRUE) / off_plays,
    
    # Penalty metrics  
    off_penalty_rate = sum(penalty, na.rm = TRUE) / off_plays,
    off_penalty_yards_per_play = sum(penalty_yards, na.rm = TRUE) / off_plays,
    
    .groups = "drop"
  )

# Enhanced defensive statistics
defensive_stats <- pbp_raw %>%
  filter(!is.na(epa), !is.na(defteam)) %>%
  group_by(season, defteam) %>%
  summarise(
    # Core defensive metrics
    def_plays = n(),
    def_epa_allowed = mean(epa, na.rm = TRUE),
    def_success_rate_allowed = mean(success, na.rm = TRUE),
    def_explosive_rate_allowed = mean(epa > 0.5, na.rm = TRUE),
    
    # Situational defense
    def_red_zone_plays = sum(yardline_100 <= 20, na.rm = TRUE),
    def_red_zone_td_allowed = mean(touchdown[yardline_100 <= 20], na.rm = TRUE),
    def_third_down_plays = sum(down == 3, na.rm = TRUE),
    def_third_down_allowed = mean(third_down_converted[down == 3], na.rm = TRUE),
    def_fourth_down_allowed = mean(fourth_down_converted[down == 4], na.rm = TRUE),
    
    # Rush defense
    def_rush_plays = sum(rush == 1, na.rm = TRUE),
    def_rush_epa_allowed = mean(epa[rush == 1], na.rm = TRUE),
    def_rush_success_allowed = mean(success[rush == 1], na.rm = TRUE),
    def_rush_yards_allowed_per_play = mean(yards_gained[rush == 1], na.rm = TRUE),
    
    # Pass defense
    def_pass_plays = sum(pass == 1, na.rm = TRUE),
    def_pass_epa_allowed = mean(epa[pass == 1], na.rm = TRUE),
    def_pass_success_allowed = mean(success[pass == 1], na.rm = TRUE),
    def_completion_rate_allowed = mean(complete_pass[pass == 1], na.rm = TRUE),
    def_pass_yards_allowed_per_play = mean(yards_gained[pass == 1], na.rm = TRUE),
    def_air_yards_allowed_per_att = mean(air_yards[pass == 1], na.rm = TRUE),
    def_yac_allowed_per_comp = mean(yards_after_catch[complete_pass == 1], na.rm = TRUE),
    
    # NEW: Defensive pressure metrics
    def_sack_rate = sum(sack, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    def_pressure_rate = sum(!is.na(qb_hit) & qb_hit == 1, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    def_qb_scramble_rate = sum(qb_scramble == 1, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    
    # NEW: Coverage metrics
    def_tight_coverage_rate = sum(complete_pass == 0 & air_yards > 0, na.rm = TRUE) / 
      sum(air_yards > 0, na.rm = TRUE),
    
    # Defensive takeaways
    def_int_rate = sum(interception, na.rm = TRUE) / sum(pass == 1, na.rm = TRUE),
    def_fumble_recovery_rate = sum(fumble_forced, na.rm = TRUE) / def_plays,
    
    .groups = "drop"
  ) %>%
  rename(posteam = defteam)

# ==============================================================================
# INJURY-ADJUSTED TEAM STATISTICS
# ==============================================================================

print("=== CALCULATING INJURY-ADJUSTED METRICS ===")

# Calculate season-long injury impact by team
team_injury_impact <- team_injury_summary %>%
  group_by(season, team) %>%
  summarise(
    # Basic injury counts
    avg_total_injuries = mean(total_injuries, na.rm = TRUE),
    avg_high_severity_injuries = mean(high_severity_injuries, na.rm = TRUE),
    avg_qb_injuries = mean(qb_injuries, na.rm = TRUE),
    avg_skill_injuries = mean(skill_injuries, na.rm = TRUE),
    avg_ol_injuries = mean(ol_injuries, na.rm = TRUE),
    
    # Injury consistency and patterns
    injury_volatility = sd(total_injuries, na.rm = TRUE),
    weeks_with_injuries = sum(total_injuries > 0, na.rm = TRUE),
    injury_consistency = 1 / (1 + sd(total_injuries, na.rm = TRUE)),
    
    # Impact scores
    avg_injury_impact_score = mean(injury_impact_score, na.rm = TRUE),
    max_injury_impact_score = max(injury_impact_score, na.rm = TRUE),
    
    # Position-specific injury rates
    qb_injury_weeks = sum(qb_injuries > 0, na.rm = TRUE),
    skill_injury_weeks = sum(skill_injuries > 0, na.rm = TRUE),
    ol_injury_weeks = sum(ol_injuries > 0, na.rm = TRUE),
    
    # NEW: Injury timing patterns
    early_season_injuries = mean(total_injuries[week <= 6], na.rm = TRUE),
    mid_season_injuries = mean(total_injuries[week > 6 & week <= 12], na.rm = TRUE), 
    late_season_injuries = mean(total_injuries[week > 12], na.rm = TRUE),
    
    .groups = "drop"
  ) %>%
  rename(posteam = team) %>%
  # Handle missing data
  mutate(across(where(is.numeric), ~ifelse(is.na(.x) | is.infinite(.x), 0, .x)))

print(paste("✓ Injury impact calculated for", nrow(team_injury_impact), "team-seasons"))

# ==============================================================================
# COMBINE ALL TEAM STATISTICS
# ==============================================================================

# Combine offensive, defensive, and injury stats
team_stats_combined <- offensive_stats %>%
  left_join(defensive_stats, by = c("season", "posteam")) %>%
  left_join(team_injury_impact, by = c("season", "posteam")) %>%
  # Calculate derived metrics including injury adjustments
  mutate(
    # Original derived metrics
    net_epa_per_play = off_epa_per_play - def_epa_allowed,
    net_success_rate = off_success_rate - def_success_rate_allowed,
    rush_pass_balance = off_rush_plays / (off_rush_plays + off_pass_plays),
    red_zone_efficiency = off_red_zone_td_rate - def_red_zone_td_allowed,
    third_down_efficiency = off_third_down_conv - def_third_down_allowed,
    turnover_margin_rate = (def_int_rate + def_fumble_recovery_rate) - (off_int_rate + off_fumble_rate),
    
    # NEW: Injury-adjusted performance metrics
    injury_adjusted_epa = net_epa_per_play * (1 - avg_injury_impact_score * 0.02), # Scale injury impact
    injury_adjusted_success = net_success_rate * (1 - avg_injury_impact_score * 0.02),
    
    # NEW: Positional injury impacts
    qb_injury_impact = avg_qb_injuries * 0.15,  # QB injuries heavily weighted
    skill_injury_impact = avg_skill_injuries * 0.08,
    ol_injury_impact = avg_ol_injuries * 0.06,
    
    # NEW: Depth and resilience metrics  
    injury_resilience = off_epa_consistency * (1 + injury_consistency),
    depth_score = 1 / (1 + avg_total_injuries),  # Teams with fewer injuries might have better depth
    
    # NEW: Pressure differential with injury context
    net_pressure_rate = def_pressure_rate - off_pressure_rate_allowed,
    injury_adjusted_protection = off_sack_rate_allowed * (1 + ol_injury_impact),
    
    # Handle any remaining NAs
    across(where(is.numeric), ~ifelse(is.na(.x) | is.infinite(.x), 0, .x))
  )

print(paste("✓ Enhanced team statistics calculated for", nrow(team_stats_enhanced), "team-seasons"))

# ==============================================================================
# RECENT FORM & MOMENTUM FEATURES
# ==============================================================================

print("=== CALCULATING RECENT FORM METRICS ===")

# Function to calculate rolling averages for recent form
calculate_rolling_stats <- function(games_df, window = 4) {
  games_df %>%
    arrange(season, week) %>%
    group_by(posteam, season) %>%
    mutate(
      # Rolling offensive performance
      rolling_off_epa = zoo::rollmean(off_epa_per_play, k = window, fill = NA, align = "right"),
      rolling_off_success = zoo::rollmean(off_success_rate, k = window, fill = NA, align = "right"),
      
      # Rolling defensive performance  
      rolling_def_epa = zoo::rollmean(def_epa_allowed, k = window, fill = NA, align = "right"),
      rolling_def_success = zoo::rollmean(def_success_rate_allowed, k = window, fill = NA, align = "right"),
      
      # Win streak indicators
      result_binary = ifelse(result > 0, 1, ifelse(result < 0, -1, 0)),
      win_streak = sequence(rle(result_binary)$lengths) * (result_binary),
      
      # Recent margin trends
      rolling_margin = zoo::rollmean(result, k = window, fill = NA, align = "right")
    ) %>%
    ungroup()
}

# Note: This would be applied to game-level data with weekly team performance
# For now, we'll create season-level features

# ==============================================================================
# DIVISION AND CONFERENCE MAPPING
# ==============================================================================

print("=== CREATING DIVISION/CONFERENCE MAPPINGS ===")

# Create division/conference mapping if not in teams data
if(!"division" %in% names(teams) || !"conference" %in% names(teams)) {
  team_divisions <- data.frame(
    team_abbr = c("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", 
                  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
                  "LV", "LAC", "LAR", "MIA", "MIN", "NE", "NO", "NYG",
                  "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"),
    division = c("NFC West", "NFC South", "AFC North", "AFC East", "NFC South", "NFC North", "AFC North", "AFC North",
                 "NFC East", "AFC West", "NFC North", "NFC North", "AFC South", "AFC South", "AFC South", "AFC West", 
                 "AFC West", "AFC West", "NFC West", "AFC East", "NFC North", "AFC East", "NFC South", "NFC East",
                 "AFC East", "NFC East", "AFC North", "NFC West", "NFC West", "NFC South", "AFC South", "NFC East"),
    conference = c("NFC", "NFC", "AFC", "AFC", "NFC", "NFC", "AFC", "AFC",
                   "NFC", "AFC", "NFC", "NFC", "AFC", "AFC", "AFC", "AFC",
                   "AFC", "AFC", "NFC", "AFC", "NFC", "AFC", "NFC", "NFC", 
                   "AFC", "NFC", "AFC", "NFC", "NFC", "NFC", "AFC", "NFC")
  )
} else {
  team_divisions <- teams %>%
    select(team_abbr, division, conference) %>%
    distinct()
}

print("✓ Division/conference mappings created")

# ==============================================================================
# MATCHUP-SPECIFIC FEATURES  
# ==============================================================================

print("=== CREATING MATCHUP FEATURES ===")

# Create game-level features with team matchups
game_features <- games_raw %>%
  filter(!is.na(result), season >= min(team_stats_combined$season)) %>%
  
  # PRESERVE ORIGINAL COLUMNS FIRST
  mutate(
    # Keep original game context variables
    original_div_game = if("div_game" %in% names(.)) div_game else NA,
    original_home_rest = if("home_rest" %in% names(.)) home_rest else NA, 
    original_away_rest = if("away_rest" %in% names(.)) away_rest else NA,
    
    # Keep core identifiers  
    game_id_orig = game_id,
    season_orig = season,
    week_orig = week,
    home_team_orig = home_team,
    away_team_orig = away_team,
    result_orig = result,
    gameday_orig = if("gameday" %in% names(.)) gameday else NA,
    gametime_orig = if("gametime" %in% names(.)) gametime else NA,
    home_score_orig = home_score,
    away_score_orig = away_score
  ) %>%
  
  # Add home team divisions/conferences
  left_join(
    team_divisions %>% rename(home_division = division, home_conference = conference),
    by = c("home_team" = "team_abbr")
  ) %>%
  
  # Add away team divisions/conferences  
  left_join(
    team_divisions %>% rename(away_division = division, away_conference = conference),
    by = c("away_team" = "team_abbr")
  ) %>%
  
  # Calculate divisional and conference games
  mutate(
    div_game = ifelse(!is.na(home_division) & !is.na(away_division), 
                      as.numeric(home_division == away_division), 
                      ifelse(!is.na(original_div_game), original_div_game, 0)),
    conference_game = ifelse(!is.na(home_conference) & !is.na(away_conference),
                             as.numeric(home_conference == away_conference), 0),
    
    # Set rest days to defaults if missing
    home_rest = ifelse(!is.na(original_home_rest), original_home_rest, 7),
    away_rest = ifelse(!is.na(original_away_rest), original_away_rest, 7)
  ) %>%
  
  # Add home team stats
  left_join(
    team_stats_combined %>% select(-contains("def_plays"), -contains("off_plays")), 
    by = c("season" = "season", "home_team" = "posteam")
  ) %>%
  rename_with(~paste0("home_", .x), .cols = matches("^(off_|def_|net_|rush_pass|red_zone|third_down|turnover_margin)")) %>%
  
  # Add away team stats
  left_join(
    team_stats_combined %>% select(-contains("def_plays"), -contains("off_plays")), 
    by = c("season" = "season", "away_team" = "posteam")  
  ) %>%
  rename_with(~paste0("away_", .x), .cols = matches("^(off_|def_|net_|rush_pass|red_zone|third_down|turnover_margin)")) %>%
  
  # Create matchup differentials
  mutate(
    # Core EPA matchups
    epa_advantage = (home_off_epa_per_play - away_def_epa_allowed) - (away_off_epa_per_play - home_def_epa_allowed),
    success_rate_advantage = (home_off_success_rate - away_def_success_rate_allowed) - (away_off_success_rate - home_def_success_rate_allowed),
    
    # Specific matchup strengths
    rush_matchup_home = home_off_rush_epa_per_play - away_def_rush_epa_allowed,
    rush_matchup_away = away_off_rush_epa_per_play - home_def_rush_epa_allowed,
    pass_matchup_home = home_off_pass_epa_per_play - away_def_pass_epa_allowed,
    pass_matchup_away = away_off_pass_epa_per_play - home_def_pass_epa_allowed,
    
    # Net matchup advantages
    rush_matchup_advantage = rush_matchup_home - rush_matchup_away,
    pass_matchup_advantage = pass_matchup_home - pass_matchup_away,
    
    # Situational matchups
    red_zone_advantage = (home_off_red_zone_td_rate - away_def_red_zone_td_allowed) - (away_off_red_zone_td_rate - home_def_red_zone_td_allowed),
    third_down_advantage = (home_off_third_down_conv - away_def_third_down_allowed) - (away_off_third_down_conv - home_def_third_down_allowed),
    
    # Turnover battle
    turnover_advantage = (home_turnover_margin_rate) - (away_turnover_margin_rate),
    
    # Style matchups
    pace_differential = home_rush_pass_balance - away_rush_pass_balance,
    pressure_differential = (home_net_pressure_rate %||% 0) - (away_net_pressure_rate %||% 0),
    
    # Game context features (now properly preserved)
    divisional_game = div_game,
    rest_differential = home_rest - away_rest,
    
    # Target variable
    home_win = as.factor(ifelse(result > 0, "Win", "Loss")),
    home_win_binary = ifelse(result > 0, 1, 0),
    point_differential = result,
    
    # Additional context
    total_score = home_score + away_score,
    home_score_pct = home_score / total_score,
    
    # Time/season features
    week_type = case_when(
      week <= 18 ~ "Regular",
      week <= 22 ~ "Playoffs", 
      TRUE ~ "Other"
    ),
    season_phase = case_when(
      week <= 6 ~ "Early",
      week <= 12 ~ "Mid",
      week <= 18 ~ "Late", 
      TRUE ~ "Playoffs"
    )
  ) %>%
  # Remove rows with missing key features
  filter(complete.cases(select(., epa_advantage, success_rate_advantage, rush_matchup_advantage, pass_matchup_advantage)))

print(paste("✓ Matchup features created for", nrow(game_features), "games"))

# Verify divisional game calculation
div_game_summary <- game_features %>%
  count(divisional_game) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

print("Divisional game distribution:")
print(div_game_summary)

# ==============================================================================
# FEATURE SELECTION & FINAL DATASET
# ==============================================================================

print("=== PREPARING FINAL MODELING DATASET ===")

# Select key features for modeling
model_features <- c(
  # Primary matchup features
  "epa_advantage",
  "success_rate_advantage",
  "rush_matchup_advantage",
  #"pass_matchup_advantage",
  
  # Situational advantages
  "red_zone_advantage",
  "third_down_advantage",
  "turnover_advantage",
  
  # Team strength features
  "home_net_epa_per_play",
  "away_net_epa_per_play",
  "home_net_success_rate",
  "away_net_success_rate",
  "pressure_differential",
  
  # Style and context
  #"pace_differential",
  "divisional_game",
  "rest_differential",
  
  # Game context
  "week", "season_phase"
)

# Create final modeling dataset
modeling_data <- game_features %>%
  select(
    # Identifiers
    game_id, season, week, home_team, away_team, gameday,
    
    # Target variables
    home_win, home_win_binary, point_differential,
    
    # Features
    all_of(model_features)
  ) %>%
  # Convert categorical features
  mutate(
    season_phase = as.factor(season_phase),
    divisional_game = as.factor(divisional_game)
  ) %>%
  # Final cleanup
  filter(complete.cases(.))

print(paste("✓ Final dataset prepared with", nrow(modeling_data), "games"))

# ==============================================================================
# SAVE PROCESSED DATA
# ==============================================================================

print("=== SAVING PROCESSED DATA ===")

# Create directory if it doesn't exist
if (!dir.exists(data_processed_path)) {
  dir.create(data_processed_path, recursive = TRUE)
}

# Save team statistics
write_rds(team_stats_combined, here(data_processed_path, "team_stats_by_season.rds"))
write_csv(team_stats_combined, here(data_processed_path, "team_stats_by_season.csv"))

# Save game features
write_rds(game_features, here(data_processed_path, "game_features_full.rds"))

# Save modeling dataset
write_rds(modeling_data, here(data_processed_path, "modeling_data.rds"))
write_csv(modeling_data, here(data_processed_path, "modeling_data.csv"))

print("✓ All processed data saved")

# ==============================================================================
# DATA SUMMARY
# ==============================================================================

print("=== FEATURE ENGINEERING SUMMARY ===")
print(paste("Processing completed at:", Sys.time()))
print(paste("Final modeling dataset:", nrow(modeling_data), "games"))
print(paste("Features available:", length(model_features)))
print(paste("Seasons covered:", min(modeling_data$season), "to", max(modeling_data$season)))

# Feature summary
cat("\nAvailable features for modeling:\n")
cat(paste("-", model_features), sep = "\n")

# Distribution of target variable
target_summary <- modeling_data %>%
  count(home_win) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))

print("Target variable distribution:")
print(target_summary)

# Basic feature statistics
print("\nKey feature statistics:")
feature_stats <- modeling_data %>%
  select(all_of(model_features[1:6])) %>%  # Show first 6 numeric features
  summarise(across(where(is.numeric), list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)))) %>%
  pivot_longer(everything(), names_to = c("feature", "stat"), names_sep = "_(?=[^_]*$)") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(across(where(is.numeric), ~round(.x, 3)))

print(feature_stats)

print("\n=== READY FOR MODEL TRAINING ===")
print("Next step: Run 03_model_training.R")

# Clean up environment
rm(pbp_raw, offensive_stats, defensive_stats, team_divisions)
gc()