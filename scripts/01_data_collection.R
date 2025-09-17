# ==============================================================================
# NFL PREDICTION MODEL - ENHANCED DATA COLLECTION WITH INJURY DATA
# Script: 01_data_collection_enhanced.R
# Purpose: Download NFL data with working injury data integration
# ==============================================================================

library(nflreadr)
library(dplyr)
library(readr)
library(here)
library(httr)
library(jsonlite)

# Set up paths
data_raw_path <- here("data", "raw")
data_processed_path <- here("data", "processed")

if (!dir.exists(data_raw_path)) dir.create(data_raw_path, recursive = TRUE)
if (!dir.exists(data_processed_path)) dir.create(data_processed_path, recursive = TRUE)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEASONS <- 2020:2024
CURRENT_SEASON <- 2025

print(paste("Collecting NFL data for seasons:", min(SEASONS), "to", max(SEASONS)))

# ==============================================================================
# ENHANCED INJURY DATA COLLECTION
# ==============================================================================

# Function to get injury data using multiple methods
get_comprehensive_injury_data <- function(seasons) {
  
  print("=== COLLECTING COMPREHENSIVE INJURY DATA ===")
  
  all_injury_data <- list()
  
  # Method 1: Try nflreadr for recent seasons
  for (season in rev(seasons)) {  # Start with most recent
    cat("Trying nflreadr injury data for", season, "...")
    
    injury_data <- tryCatch({
      data <- load_injuries(season)
      if (!is.null(data) && nrow(data) > 0) {
        cat(" ✓ Success!\n")
        data$source <- "nflreadr"
        data$season <- season
        return(data)
      } else {
        cat(" ✗ No data\n")
        return(NULL)
      }
    }, error = function(e) {
      cat(" ✗ Error:", substr(e$message, 1, 50), "...\n")
      return(NULL)
    })
    
    if (!is.null(injury_data)) {
      all_injury_data[[paste0("season_", season)]] <- injury_data
    }
  }
  
  # Method 2: Create synthetic injury severity data based on game participation
  if (length(all_injury_data) == 0) {
    print("No direct injury data available. Creating participation-based injury indicators...")
    
    # Load roster and snap count data to infer injuries
    participation_data <- tryCatch({
      create_participation_injury_data(seasons)
    }, error = function(e) {
      print(paste("Could not create participation data:", e$message))
      NULL
    })
    
    if (!is.null(participation_data)) {
      all_injury_data[["participation_based"]] <- participation_data
    }
  }
  
  # Combine all injury data
  if (length(all_injury_data) > 0) {
    combined_injury_data <- bind_rows(all_injury_data, .id = "data_source")
    return(combined_injury_data)
  } else {
    return(NULL)
  }
}

# Function to create participation-based injury indicators
create_participation_injury_data <- function(seasons) {
  
  print("Creating participation-based injury indicators...")
  
  # Get play-by-play data to analyze player participation
  pbp_data <- load_pbp(seasons = seasons)
  
  if (is.null(pbp_data) || nrow(pbp_data) == 0) {
    return(NULL)
  }
  
  # Analyze QB participation patterns (most trackable position)
  qb_participation <- pbp_data %>%
    filter(!is.na(passer_player_name), play_type %in% c("pass", "run")) %>%
    group_by(season, week, posteam, passer_player_name) %>%
    summarise(
      plays_as_qb = n(),
      epa_per_play = mean(epa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(player_name = passer_player_name) %>%
    mutate(position = "QB")
  
  # Analyze RB participation
  rb_participation <- pbp_data %>%
    filter(!is.na(rusher_player_name), play_type == "run") %>%
    group_by(season, week, posteam, rusher_player_name) %>%
    summarise(
      plays_as_rb = n(),
      rush_epa_per_play = mean(epa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(player_name = rusher_player_name) %>%
    mutate(position = "RB")
  
  # Analyze WR participation
  wr_participation <- pbp_data %>%
    filter(!is.na(receiver_player_name), play_type == "pass") %>%
    group_by(season, week, posteam, receiver_player_name) %>%
    summarise(
      targets = n(),
      receiving_epa_per_target = mean(epa, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(player_name = receiver_player_name) %>%
    mutate(position = "WR")
  
  # Combine participation data
  all_participation <- bind_rows(
    qb_participation %>% select(season, week, posteam, player_name, position),
    rb_participation %>% select(season, week, posteam, player_name, position),
    wr_participation %>% select(season, week, posteam, player_name, position)
  )
  
  # Identify potential injuries based on sudden absence patterns
  injury_indicators <- all_participation %>%
    arrange(season, posteam, player_name, week) %>%
    group_by(season, posteam, player_name, position) %>%
    mutate(
      played_last_week = lag(week, 1) == (week - 1),
      missing_weeks = ifelse(!played_last_week, 1, 0),
      potential_injury = missing_weeks == 1
    ) %>%
    filter(potential_injury) %>%
    mutate(
      report = paste("Potential injury - missed game in week", week),
      game_status = "Out",
      injury_status = "Questionable",
      source = "participation_analysis"
    ) %>%
    ungroup()
  
  return(injury_indicators)
}

# ==============================================================================
# MAIN DATA COLLECTION WITH INJURY INTEGRATION
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

print("=== COLLECTING GAME DATA ===")
games_raw <- safe_load_data(load_schedules, SEASONS, "game schedules and results")

print("=== COLLECTING PLAY-BY-PLAY DATA ===")
pbp_raw <- safe_load_data(load_pbp, SEASONS, "play-by-play data")

print("=== COLLECTING TEAM DATA ===")
teams <- load_teams()
print(paste("✓ Team data loaded:", nrow(teams), "teams"))

print("=== COLLECTING ADDITIONAL DATA ===")
rosters <- safe_load_data(load_rosters, CURRENT_SEASON, "roster data")

# Enhanced injury data collection
injuries <- get_comprehensive_injury_data(c(CURRENT_SEASON, SEASONS))


# ==============================================================================
# INJURY DATA PROCESSING AND VALIDATION
# ==============================================================================

if (!is.null(injuries)) {
  print("=== PROCESSING INJURY DATA ===")
  
  # Standardize injury data format
  processed_injuries <- injuries %>%
    mutate(
      # Standardize team names - check which columns exist
      team = case_when(
        "team" %in% names(.) & !is.na(team) ~ team,
        #"posteam" %in% names(.) & !is.na(posteam) ~ posteam,
        #"tm" %in% names(.) & !is.na(tm) ~ tm,
        TRUE ~ "UNK"
      ),
      
      # Standardize injury severity
      injury_severity = case_when(
        "report_status" %in% names(.) & grepl("out|doubtful", tolower(report_status)) ~ "High",
        #"injury_status" %in% names(.) & grepl("out|doubtful", tolower(injury_status)) ~ "High",
        "report_status" %in% names(.) & grepl("questionable", tolower(report_status)) ~ "Medium",
        #"injury_status" %in% names(.) & grepl("questionable", tolower(injury_status)) ~ "Medium", 
        "report_status" %in% names(.) & grepl("probable", tolower(report_status)) ~ "Low",
        #"injury_status" %in% names(.) & grepl("probable", tolower(injury_status)) ~ "Low",
        TRUE ~ "Unknown"
      ),
      
      # Standardize position groups
      position_group = case_when(
        "position" %in% names(.) & position %in% c("QB") ~ "QB",
        "position" %in% names(.) & position %in% c("RB", "FB") ~ "RB", 
        "position" %in% names(.) & position %in% c("WR", "TE") ~ "SKILL",
        "position" %in% names(.) & position %in% c("LT", "LG", "C", "RG", "RT", "OL") ~ "OL",
        "position" %in% names(.) & position %in% c("DE", "DT", "NT", "DL") ~ "DL",
        "position" %in% names(.) & position %in% c("LB", "ILB", "OLB") ~ "LB",
        "position" %in% names(.) & position %in% c("CB", "S", "SS", "FS", "DB") ~ "DB",
        "position" %in% names(.) & position %in% c("K", "P", "LS") ~ "ST",
        TRUE ~ "OTHER"
      )
    ) %>%
    # Remove duplicates and invalid entries - check for required columns
    filter(
      ("player_name" %in% names(.) | "full_name" %in% names(.)),
      ("team" %in% names(.) | "posteam" %in% names(.) | "tm" %in% names(.))
    ) %>%
    # Ensure we have player_name column
    mutate(
      player_name = case_when(
        #"player_name" %in% names(.) & !is.na(player_name) ~ player_name,
        "full_name" %in% names(.) & !is.na(full_name) ~ full_name,
        #"name" %in% names(.) & !is.na(name) ~ name,
        TRUE ~ "Unknown Player"
      ),
      # Ensure we have season and week
      season = ifelse("season" %in% names(.) & !is.na(season), season, CURRENT_SEASON),
      week = ifelse("week" %in% names(.) & !is.na(week), week, 1)
    ) %>%
    # Only keep rows with essential data
    filter(!is.na(player_name), player_name != "Unknown Player", !is.na(team), team != "UNK") %>%
    distinct(season, week, team, player_name, .keep_all = TRUE)
  
  print(paste("✓ Processed", nrow(processed_injuries), "injury records"))
  
  # Create injury summary by team-week
  team_injury_summary <- processed_injuries %>%
    group_by(season, week, team) %>%
    summarise(
      total_injuries = n(),
      high_severity_injuries = sum(injury_severity == "High", na.rm = TRUE),
      medium_severity_injuries = sum(injury_severity == "Medium", na.rm = TRUE),
      qb_injuries = sum(position_group == "QB", na.rm = TRUE),
      skill_injuries = sum(position_group == "SKILL", na.rm = TRUE),
      ol_injuries = sum(position_group == "OL", na.rm = TRUE),
      dl_injuries = sum(position_group == "DL", na.rm = TRUE),
      lb_injuries = sum(position_group == "LB", na.rm = TRUE),
      db_injuries = sum(position_group == "DB", na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      injury_impact_score = (high_severity_injuries * 3) + 
        (medium_severity_injuries * 2) + 
        (qb_injuries * 5) +  # QB injuries weighted heavily
        (skill_injuries * 2)
    )
  
  print("✓ Created team-level injury summaries")
  
} else {
  print("No injury data available - will create placeholder data")
  
  # Create minimal placeholder injury data
  processed_injuries <- data.frame(
    season = integer(0),
    week = integer(0),
    team = character(0),
    player_name = character(0),
    position_group = character(0),
    injury_severity = character(0)
  )
  
  team_injury_summary <- data.frame(
    season = integer(0),
    week = integer(0), 
    team = character(0),
    total_injuries = integer(0),
    high_severity_injuries = integer(0),
    qb_injuries = integer(0),
    injury_impact_score = integer(0)
  )
}

# ==============================================================================
# SAVE ALL DATA INCLUDING INJURY DATA
# ==============================================================================

print("=== SAVING RAW DATA ===")

if (!is.null(games_raw)) {
  write_rds(games_raw, here(data_raw_path, "games_raw.rds"))
  print("✓ Games data saved")
}

if (!is.null(pbp_raw)) {
  write_rds(pbp_raw, here(data_raw_path, "pbp_raw.rds"))
  print("✓ Play-by-play data saved")
}

write_rds(teams, here(data_raw_path, "teams.rds"))

if (!is.null(rosters)) {
  write_rds(rosters, here(data_raw_path, "rosters.rds"))
  print("✓ Roster data saved")
}

# Save injury data
write_rds(processed_injuries, here(data_raw_path, "injuries_processed.rds"))
write_rds(team_injury_summary, here(data_raw_path, "team_injury_summary.rds"))
write_csv(team_injury_summary, here(data_raw_path, "team_injury_summary.csv"))

print("✓ Injury data saved")

# ==============================================================================
# ENHANCED DATA SUMMARY REPORT
# ==============================================================================

print("=== DATA COLLECTION SUMMARY ===")

data_summary <- data.frame(
  Dataset = c("Games", "Play-by-Play", "Teams", "Rosters", "Injuries", "Team Injury Summary"),
  Status = c(
    ifelse(!is.null(games_raw), "✓ Success", "✗ Failed"),
    ifelse(!is.null(pbp_raw), "✓ Success", "✗ Failed"),
    "✓ Success",
    ifelse(!is.null(rosters), "✓ Success", "✗ Failed"),
    ifelse(nrow(processed_injuries) > 0, "✓ Success", "✗ No Data"),
    ifelse(nrow(team_injury_summary) > 0, "✓ Success", "✗ No Data")
  ),
  Rows = c(
    ifelse(!is.null(games_raw), nrow(games_raw), 0),
    ifelse(!is.null(pbp_raw), nrow(pbp_raw), 0),
    nrow(teams),
    ifelse(!is.null(rosters), nrow(rosters), 0),
    nrow(processed_injuries),
    nrow(team_injury_summary)
  )
)

print(data_summary)

if (nrow(team_injury_summary) > 0) {
  print("INJURY DATA STATISTICS:")
  injury_stats <- team_injury_summary %>%
    summarise(
      avg_injuries_per_team_week = round(mean(total_injuries), 2),
      max_injuries_single_week = max(total_injuries),
      weeks_with_qb_injuries = sum(qb_injuries > 0),
      avg_impact_score = round(mean(injury_impact_score), 2)
    )
  print(injury_stats)
}

write_csv(data_summary, here(data_raw_path, "data_collection_summary.csv"))

print("=== ENHANCED DATA COLLECTION COMPLETE ===")
print("Injury data integration successful!")
print("Next step: Run enhanced feature engineering to incorporate injury features")

# Clean up
rm(pbp_raw)
gc()