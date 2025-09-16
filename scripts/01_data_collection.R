# ==============================================================================
# NFL PREDICTION MODEL - DATA COLLECTION
# Script: 01_data_collection.R
# Purpose: Download and prepare raw NFL data using nflreadr
# ==============================================================================

# Load required libraries
library(nflreadr)
library(dplyr)
library(readr)
library(here)

# Set up paths using here() for cross-platform compatibility
data_raw_path <- here("data", "raw")
data_processed_path <- here("data", "processed")

# Create directories if they don't exist
if (!dir.exists(data_raw_path)) dir.create(data_raw_path, recursive = TRUE)
if (!dir.exists(data_processed_path)) dir.create(data_processed_path, recursive = TRUE)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define seasons to collect (adjust as needed)
SEASONS <- 2020:2024  # Changed from 2019:2023 (shift up 1 year)
CURRENT_SEASON <- 2025  # Changed from 2024

print(paste("Collecting NFL data for seasons:", min(SEASONS), "to", max(SEASONS)))

# ==============================================================================
# DATA COLLECTION FUNCTIONS
# ==============================================================================

# Function to safely load data with error handling
safe_load_data <- function(load_function, seasons, description) {
  cat("Loading", description, "...")
  
  tryCatch({
    data <- load_function(seasons = seasons)
    cat(" ✓ Success!", nrow(data), "rows\n")
    return(data)
  }, error = function(e) {
    cat(" ✗ Error:", e$message, "\n")
    return(NULL)
  })
}

# ==============================================================================
# COLLECT RAW DATA
# ==============================================================================

# 1. Game schedules and results
print("=== COLLECTING GAME DATA ===")
games_raw <- safe_load_data(
  load_schedules, 
  SEASONS, 
  "game schedules and results"
)

# 2. Play-by-play data (this may take a while!)
print("=== COLLECTING PLAY-BY-PLAY DATA ===")
print("Warning: This may take several minutes for multiple seasons...")
pbp_raw <- safe_load_data(
  load_pbp,
  SEASONS,
  "play-by-play data"
)

# 3. Team information
print("=== COLLECTING TEAM DATA ===")
teams <- load_teams()
print(paste("✓ Team data loaded:", nrow(teams), "teams"))

# 4. Additional useful datasets
print("=== COLLECTING ADDITIONAL DATA ===")

# Roster data (for current season)
rosters <- safe_load_data(
  load_rosters,
  CURRENT_SEASON,
  "roster data"
)

# Injuries (current season only - historical not available)
# Replace the injury loading section in 01_data_collection.R with this:

# Injuries (load most recent available season)
print("=== COLLECTING INJURY DATA ===")
injuries <- NULL
injury_seasons_to_try <- c(CURRENT_SEASON, CURRENT_SEASON - 1, CURRENT_SEASON - 2)

for (injury_season in injury_seasons_to_try) {
  cat("Trying injury data for season", injury_season, "...")
  injuries <- tryCatch({
    data <- load_injuries(injury_season)
    cat(" ✓ Success!\n")
    data$season_loaded <- injury_season  # Track which season we got
    return(data)
  }, error = function(e) {
    cat(" ✗ Not available\n")
    return(NULL)
  })
  
  if (!is.null(injuries)) break
}

if (is.null(injuries)) {
  print("No injury data available for recent seasons")
} else {
  print(paste("✓ Injury data loaded for season", unique(injuries$season_loaded)))
}

# ==============================================================================
# BASIC DATA VALIDATION
# ==============================================================================

print("=== DATA VALIDATION ===")

if (!is.null(games_raw)) {
  print(paste("Games data:", nrow(games_raw), "games"))
  print(paste("Seasons covered:", min(games_raw$season, na.rm = TRUE), "to", max(games_raw$season, na.rm = TRUE)))
  print(paste("Games with results:", sum(!is.na(games_raw$result))))
}

if (!is.null(pbp_raw)) {
  print(paste("Play-by-play data:", nrow(pbp_raw), "plays"))
  print(paste("Seasons covered:", min(pbp_raw$season, na.rm = TRUE), "to", max(pbp_raw$season, na.rm = TRUE)))
}

# Check for any obvious data issues
if (!is.null(games_raw) && !is.null(pbp_raw)) {
  games_per_season <- games_raw %>% 
    count(season) %>% 
    arrange(season)
  print("Games per season:")
  print(games_per_season)
}

# ==============================================================================
# SAVE RAW DATA
# ==============================================================================

print("=== SAVING RAW DATA ===")

# Save main datasets
if (!is.null(games_raw)) {
  write_rds(games_raw, here(data_raw_path, "games_raw.rds"))
  print("✓ Games data saved")
}

if (!is.null(pbp_raw)) {
  write_rds(pbp_raw, here(data_raw_path, "pbp_raw.rds"))
  print("✓ Play-by-play data saved")
}

write_rds(teams, here(data_raw_path, "teams.rds"))
print("✓ Teams data saved")

if (!is.null(rosters)) {
  write_rds(rosters, here(data_raw_path, "rosters.rds"))
  print("✓ Roster data saved")
}

if (!is.null(injuries)) {
  write_rds(injuries, here(data_raw_path, "injuries.rds"))
  print("✓ Injury data saved")
}

# ==============================================================================
# DATA SUMMARY REPORT
# ==============================================================================

print("=== DATA COLLECTION SUMMARY ===")
print(paste("Data collection completed at:", Sys.time()))
print(paste("Files saved to:", data_raw_path))

# Create a simple summary
data_summary <- data.frame(
  Dataset = c("Games", "Play-by-Play", "Teams", "Rosters", "Injuries"),
  Status = c(
    ifelse(!is.null(games_raw), "✓ Success", "✗ Failed"),
    ifelse(!is.null(pbp_raw), "✓ Success", "✗ Failed"),
    "✓ Success",
    ifelse(!is.null(rosters), "✓ Success", "✗ Failed"),
    ifelse(!is.null(injuries), "✓ Success", "Not Available")
  ),
  Rows = c(
    ifelse(!is.null(games_raw), nrow(games_raw), 0),
    ifelse(!is.null(pbp_raw), nrow(pbp_raw), 0),
    nrow(teams),
    ifelse(!is.null(rosters), nrow(rosters), 0),
    ifelse(!is.null(injuries), nrow(injuries), 0)
  )
)

print(data_summary)

# Save summary report
write_csv(data_summary, here(data_raw_path, "data_collection_summary.csv"))

print("=== READY FOR FEATURE ENGINEERING ===")
print("Next step: Run 02_feature_engineering.R")

# ==============================================================================
# MEMORY CLEANUP
# ==============================================================================

# Clean up large objects to free memory
# (Keep this commented out if you want to explore the data interactively)
# rm(pbp_raw, games_raw)
# gc()  # Garbage collection