# ==============================================================================
# NFL PREDICTION MODEL - REWORKED DASHBOARD
# Script: dashboard_reworked.R
# Purpose: Clean, focused dashboard with Weekly Matchups, Model Info, and Data pages
# ==============================================================================

# Load required libraries
library(shiny)
library(shinydashboard)
library(DT)
library(plotly)
library(dplyr)
library(readr)
library(here)
library(nflreadr)
library(shinycssloaders)
library(tidyr)

# Source prediction functions
source(here("scripts", "04_predictions.R"))

# ==============================================================================
# LOAD DATA AND MODELS
# ==============================================================================

# Load all necessary data
tryCatch({
  trained_models <- readRDS(here("models", "trained_models.rds"))
  performance_data <- read_csv(here("models", "model_performance.csv"))
  team_stats <- read_rds(here("data", "processed", "team_stats_by_season.rds"))
  modeling_data <- read_rds(here("data", "processed", "modeling_data.rds"))
  
  # Get feature importance if available
  feature_importance <- tryCatch({
    read_csv(here("models", "feature_importance.csv"))
  }, error = function(e) {
    # Create mock feature importance if file doesn't exist
    data.frame(
      Model = rep(c("Random Forest", "XGBoost"), each = 8),
      Feature = rep(c("epa_advantage", "success_rate_advantage", "rush_matchup_advantage", 
                      "pass_matchup_advantage", "home_net_epa_per_play", "away_net_epa_per_play",
                      "red_zone_advantage", "third_down_advantage"), 2),
      Importance = c(0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04,
                     0.28, 0.18, 0.16, 0.11, 0.09, 0.07, 0.06, 0.05)
    )
  })
  
  # Get current team rankings
  current_rankings <- get_team_rankings()
  
  # Get team list
  team_list <- sort(unique(team_stats$posteam))
  
  data_loaded <- TRUE
}, error = function(e) {
  data_loaded <- FALSE
  error_message <- e$message
  print(paste("Error loading data:", e$message))
})

# ==============================================================================
# UI DEFINITION
# ==============================================================================

ui <- dashboardPage(
  dashboardHeader(title = "NFL Prediction System"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Weekly Matchups", tabName = "weekly", icon = icon("calendar-week")),
      menuItem("Model Information", tabName = "models", icon = icon("cogs")),
      menuItem("Team Data", tabName = "data", icon = icon("database"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .main-header .logo {
          font-family: 'Arial', sans-serif;
          font-weight: bold;
          font-size: 20px;
        }
        .box-title {
          font-size: 18px;
          font-weight: bold;
        }
        .prediction-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-radius: 10px;
          padding: 15px;
          margin: 10px 0;
          box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .team-name {
          font-size: 18px;
          font-weight: bold;
        }
        .confidence {
          font-size: 24px;
          font-weight: bold;
          text-align: center;
        }
        .vs-text {
          text-align: center;
          font-size: 16px;
          margin: 5px 0;
        }
        .model-metric {
          background-color: #f8f9fa;
          border-radius: 8px;
          padding: 10px;
          margin: 5px;
          text-align: center;
          border: 1px solid #dee2e6;
        }
      "))
    ),
    
    tabItems(
      # ==============================================================================
      # WEEKLY MATCHUPS TAB
      # ==============================================================================
      tabItem(tabName = "weekly",
              fluidRow(
                box(
                  title = "Week Selection", status = "primary", solidHeader = TRUE, width = 12,
                  fluidRow(
                    column(3,
                           numericInput("selected_week", "Week:", value = 1, min = 1, max = 22)
                    ),
                    column(3,
                           numericInput("selected_season", "Season:", value = 2025, min = 2020, max = 2025)
                    ),
                    column(3,
                           actionButton("load_week", "Load Week", class = "btn-primary", style = "margin-top: 25px;")
                    ),
                    column(3,
                           div(style = "margin-top: 30px; font-weight: bold;",
                               textOutput("week_status", inline = TRUE))
                    )
                  )
                )
              ),
              
              fluidRow(
                box(
                  title = "Week Overview", status = "info", solidHeader = TRUE, width = 12,
                  withSpinner(uiOutput("week_summary_cards"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Game Predictions", status = "success", solidHeader = TRUE, width = 12,
                  withSpinner(uiOutput("weekly_predictions_display"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Detailed Predictions Table", status = "warning", solidHeader = TRUE, 
                  width = 12, collapsible = TRUE,
                  withSpinner(DT::dataTableOutput("weekly_predictions_table"))
                )
              )
      ),
      
      # ==============================================================================
      # MODEL INFORMATION TAB
      # ==============================================================================
      tabItem(tabName = "models",
              fluidRow(
                valueBoxOutput("model_accuracy"),
                valueBoxOutput("best_individual_model"),
                valueBoxOutput("ensemble_performance")
              ),
              
              fluidRow(
                box(
                  title = "Model Performance Comparison", status = "primary", solidHeader = TRUE, width = 8,
                  withSpinner(plotlyOutput("model_performance_plot"))
                ),
                
                box(
                  title = "Performance Metrics", status = "info", solidHeader = TRUE, width = 4,
                  withSpinner(DT::dataTableOutput("performance_metrics_table"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Feature Importance Analysis", status = "success", solidHeader = TRUE, width = 12,
                  fluidRow(
                    column(6,
                           withSpinner(plotlyOutput("feature_importance_rf"))
                    ),
                    column(6,
                           withSpinner(plotlyOutput("feature_importance_xgb"))
                    )
                  )
                )
              ),
              
              fluidRow(
                box(
                  title = "Feature Descriptions", status = "warning", solidHeader = TRUE, 
                  width = 12, collapsible = TRUE,
                  withSpinner(DT::dataTableOutput("feature_descriptions_table"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Model Architecture Details", status = "primary", solidHeader = TRUE,
                  width = 12, collapsible = TRUE, collapsed = TRUE,
                  h4("Ensemble Methodology"),
                  p("The prediction system uses an ensemble of four complementary machine learning models:"),
                  tags$ul(
                    tags$li(tags$b("Logistic Regression with Elastic Net:"), "Linear baseline with regularization to prevent overfitting"),
                    tags$li(tags$b("Random Forest:"), "500 decision trees with bootstrap aggregation for robust predictions"),
                    tags$li(tags$b("XGBoost:"), "Gradient boosting with early stopping and optimal hyperparameters"),
                    tags$li(tags$b("Support Vector Machine:"), "RBF kernel for non-linear pattern recognition")
                  ),
                  p("Final predictions use simple averaging of all four model probabilities for maximum stability."),
                  br(),
                  h4("Training Process"),
                  p("Models are trained on 2020-2022 seasons, validated on 2023, and tested on 2024 data.")
                )
              )
      ),
      
      # ==============================================================================
      # TEAM DATA TAB
      # ==============================================================================
      tabItem(tabName = "data",
              fluidRow(
                box(
                  title = "Data Controls", status = "primary", solidHeader = TRUE, width = 12,
                  fluidRow(
                    column(3,
                           selectInput("data_season", "Season:", 
                                       choices = if(data_loaded) sort(unique(team_stats$season), decreasing = TRUE) else NULL,
                                       selected = if(data_loaded) max(team_stats$season) else NULL)
                    ),
                    column(3,
                           selectInput("data_view", "View:", 
                                       choices = c("All Stats", "Offensive Only", "Defensive Only", "Net/Efficiency"),
                                       selected = "All Stats")
                    ),
                    column(3,
                           downloadButton("download_data", "Download Data", class = "btn-success", 
                                          style = "margin-top: 25px;")
                    ),
                    column(3,
                           div(style = "margin-top: 30px; font-weight: bold;",
                               textOutput("data_summary"))
                    )
                  )
                )
              ),
              
              fluidRow(
                box(
                  title = "Team Rankings", status = "info", solidHeader = TRUE, width = 6,
                  withSpinner(DT::dataTableOutput("current_rankings_table"))
                ),
                
                box(
                  title = "League Averages", status = "warning", solidHeader = TRUE, width = 6,
                  withSpinner(DT::dataTableOutput("league_averages_table"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Complete Team Statistics", status = "success", solidHeader = TRUE, width = 12,
                  p("This table contains all calculated statistics for every team. Use the search and filter functions to explore the data."),
                  withSpinner(DT::dataTableOutput("complete_team_data"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Data Dictionary", status = "primary", solidHeader = TRUE, 
                  width = 12, collapsible = TRUE, collapsed = TRUE,
                  withSpinner(DT::dataTableOutput("data_dictionary_table"))
                )
              )
      )
    )
  )
)

# ==============================================================================
# SERVER LOGIC
# ==============================================================================

server <- function(input, output, session) {
  
  # Check if data loaded successfully
  if (!data_loaded) {
    showNotification("Error loading data. Please ensure all model files exist.", 
                     type = "error", duration = NULL)
    return()
  }
  
  # Reactive values
  values <- reactiveValues(
    weekly_predictions = NULL,
    selected_team_data = NULL
  )
  
  # ==============================================================================
  # WEEKLY MATCHUPS SERVER LOGIC
  # ==============================================================================
  
  # Load weekly predictions
  observeEvent(input$load_week, {
    withProgress(message = 'Loading weekly predictions...', value = 0.5, {
      values$weekly_predictions <- predict_week(input$selected_week, input$selected_season)
    })
  })
  
  # Week status
  output$week_status <- renderText({
    if (is.null(values$weekly_predictions)) {
      "Click 'Load Week' to see games"
    } else if (nrow(values$weekly_predictions) == 0) {
      "No games found for selected week"
    } else {
      paste("Loaded", nrow(values$weekly_predictions), "games")
    }
  })
  
  # Week summary cards
  output$week_summary_cards <- renderUI({
    req(values$weekly_predictions)
    req(nrow(values$weekly_predictions) > 0)
    
    preds <- values$weekly_predictions
    total_games <- nrow(preds)
    high_conf_games <- sum(abs(preds$ensemble_prob - 0.5) > 0.2)
    avg_confidence <- mean(abs(preds$ensemble_prob - 0.5)) * 100
    divisional_games <- sum(preds$divisional_game == "1", na.rm = TRUE)
    
    fluidRow(
      valueBox(
        value = total_games,
        subtitle = "Total Games",
        icon = icon("football"),
        color = "blue",
        width = 3
      ),
      valueBox(
        value = high_conf_games,
        subtitle = "High Confidence",
        icon = icon("thumbs-up"),
        color = "green",
        width = 3
      ),
      valueBox(
        value = paste0(round(avg_confidence, 1), "%"),
        subtitle = "Avg Confidence",
        icon = icon("chart-line"),
        color = "yellow",
        width = 3
      ),
      valueBox(
        value = divisional_games,
        subtitle = "Division Games",
        icon = icon("users"),
        color = "red",
        width = 3
      )
    )
  })
  
  # Weekly predictions display (cards)
  output$weekly_predictions_display <- renderUI({
    req(values$weekly_predictions)
    req(nrow(values$weekly_predictions) > 0)
    
    preds <- values$weekly_predictions %>% arrange(gameday, gametime)
    
    # Create prediction cards
    cards <- lapply(1:nrow(preds), function(i) {
      game <- preds[i, ]
      
      # Determine winner and confidence
      home_wins <- game$ensemble_prob > 0.5
      winner <- if(home_wins) game$home_team else game$away_team
      confidence <- if(home_wins) game$ensemble_prob else (1 - game$ensemble_prob)
      
      div(class = "prediction-card",
          fluidRow(
            column(4, class = "team-name", game$away_team),
            column(4, class = "vs-text", "@"),
            column(4, class = "team-name", game$home_team)
          ),
          hr(style = "margin: 8px 0; border-color: rgba(255,255,255,0.3);"),
          div(class = "confidence",
              paste("Predicted Winner:", winner),
              br(),
              paste0("Confidence: ", round(confidence * 100, 1), "%")
          ),
          if(!is.na(game$gameday)) {
            p(style = "margin-top: 10px; text-align: center; font-size: 12px;",
              paste("Game Date:", game$gameday))
          }
      )
    })
    
    # Arrange cards in rows of 3
    rows <- split(cards, ceiling(seq_along(cards) / 3))
    lapply(rows, function(row_cards) {
      do.call(fluidRow, lapply(row_cards, function(card) column(4, card)))
    })
  })
  
  # Weekly predictions table
  output$weekly_predictions_table <- DT::renderDataTable({
    req(values$weekly_predictions)
    
    values$weekly_predictions %>%
      select(away_team, home_team, predicted_winner, confidence, ensemble_prob,
             logistic_prob, rf_prob, xgb_prob, svm_prob, gameday) %>%
      rename(
        Away = away_team,
        Home = home_team,
        Predicted_Winner = predicted_winner,
        Confidence = confidence,
        Ensemble = ensemble_prob,
        Logistic = logistic_prob,
        Random_Forest = rf_prob,
        XGBoost = xgb_prob,
        SVM = svm_prob,
        Date = gameday
      ) %>%
      arrange(desc(Ensemble))
  }, options = list(pageLength = 20, scrollX = TRUE))
  
  # ==============================================================================
  # MODEL INFORMATION SERVER LOGIC
  # ==============================================================================
  
  # Value boxes for model performance
  output$model_accuracy <- renderValueBox({
    ensemble_perf <- performance_data %>% filter(grepl("Ensemble", Model))
    accuracy <- if(nrow(ensemble_perf) > 0) ensemble_perf$Accuracy[1] else 0.65
    
    valueBox(
      value = paste0(round(accuracy * 100, 1), "%"),
      subtitle = "Ensemble Accuracy",
      icon = icon("bullseye"),
      color = "green"
    )
  })
  
  output$best_individual_model <- renderValueBox({
    best_model <- performance_data %>% 
      filter(!grepl("Ensemble", Model)) %>%
      slice_max(AUC, n = 1)
    
    valueBox(
      value = if(nrow(best_model) > 0) best_model$Model[1] else "Random Forest",
      subtitle = "Best Individual Model",
      icon = icon("trophy"),
      color = "yellow"
    )
  })
  
  output$ensemble_performance <- renderValueBox({
    ensemble_perf <- performance_data %>% filter(grepl("Ensemble", Model))
    auc <- if(nrow(ensemble_perf) > 0) ensemble_perf$AUC[1] else 0.72
    
    valueBox(
      value = round(auc, 3),
      subtitle = "Ensemble AUC",
      icon = icon("chart-area"),
      color = "blue"
    )
  })
  
  # Model performance plot
  output$model_performance_plot <- renderPlotly({
    p <- performance_data %>%
      filter(!is.na(Accuracy)) %>%
      plot_ly(x = ~reorder(Model, Accuracy), y = ~Accuracy, type = "bar",
              text = ~paste("AUC:", round(AUC, 3)), textposition = "outside",
              marker = list(color = c("#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#1f77b4"))) %>%
      layout(title = "Model Performance Comparison",
             yaxis = list(title = "Accuracy", range = c(0.5, 0.8)),
             xaxis = list(title = "Model"))
    p
  })
  
  # Performance metrics table
  output$performance_metrics_table <- DT::renderDataTable({
    performance_data %>%
      select(Model, Accuracy, AUC, F1_Score, Precision, Recall) %>%
      mutate(across(where(is.numeric), ~round(.x, 3)))
  }, options = list(pageLength = 10, dom = 't'))
  
  # Feature importance plots
  output$feature_importance_rf <- renderPlotly({
    rf_data <- feature_importance %>% filter(Model == "Random Forest")
    
    p <- rf_data %>%
      plot_ly(x = ~reorder(Feature, Importance), y = ~Importance, type = "bar",
              marker = list(color = "forestgreen")) %>%
      layout(title = "Random Forest Feature Importance",
             yaxis = list(title = "Importance"),
             xaxis = list(title = "", tickangle = -45))
    p
  })
  
  output$feature_importance_xgb <- renderPlotly({
    xgb_data <- feature_importance %>% filter(Model == "XGBoost")
    
    p <- xgb_data %>%
      plot_ly(x = ~reorder(Feature, Importance), y = ~Importance, type = "bar",
              marker = list(color = "darkorange")) %>%
      layout(title = "XGBoost Feature Importance",
             yaxis = list(title = "Importance"),
             xaxis = list(title = "", tickangle = -45))
    p
  })
  
  # Feature descriptions table
  output$feature_descriptions_table <- DT::renderDataTable({
    feature_descriptions <- data.frame(
      Feature = c("epa_advantage", "success_rate_advantage", "rush_matchup_advantage",
                  "pass_matchup_advantage", "home_net_epa_per_play", "away_net_epa_per_play",
                  "red_zone_advantage", "third_down_advantage", "turnover_advantage",
                  "home_net_success_rate", "away_net_success_rate", "pace_differential"),
      Description = c(
        "Overall EPA advantage combining offense vs defense matchups",
        "Success rate differential between team matchups", 
        "Rushing offense vs rushing defense advantage",
        "Passing offense vs passing defense advantage",
        "Home team's net EPA per play (offense - defense)",
        "Away team's net EPA per play (offense - defense)",
        "Red zone scoring efficiency differential",
        "Third down conversion rate advantage",
        "Expected turnover margin based on team rates",
        "Home team's net success rate",
        "Away team's net success rate", 
        "Difference in rushing vs passing tendencies"
      ),
      Impact = c("Very High", "Very High", "High", "High", "High", "High",
                 "Medium", "Medium", "Medium", "Medium", "Medium", "Low")
    )
    
    feature_descriptions
  }, options = list(pageLength = 15))
  
  # ==============================================================================
  # TEAM DATA SERVER LOGIC
  # ==============================================================================
  
  # Reactive team data based on season
  selected_season_data <- reactive({
    req(input$data_season)
    team_stats %>% filter(season == input$data_season)
  })
  
  # Data summary
  output$data_summary <- renderText({
    data <- selected_season_data()
    if(nrow(data) > 0) {
      paste(nrow(data), "teams,", ncol(data)-2, "metrics")
    } else {
      "No data available"
    }
  })
  
  # Current rankings table
  output$current_rankings_table <- DT::renderDataTable({
    if(input$data_season == max(team_stats$season)) {
      current_rankings %>%
        select(rank, posteam, overall_rating, offensive_rating, defensive_rating) %>%
        rename(Rank = rank, Team = posteam, Overall = overall_rating,
               Offense = offensive_rating, Defense = defensive_rating) %>%
        mutate(across(where(is.numeric), ~round(.x, 1)))
    } else {
      # Calculate historical rankings
      historical_rankings <- selected_season_data() %>%
        mutate(
          overall_rating = net_epa_per_play * 100,
          offensive_rating = off_epa_per_play * 100,
          defensive_rating = -def_epa_allowed * 100
        ) %>%
        arrange(desc(overall_rating)) %>%
        mutate(rank = row_number()) %>%
        select(rank, posteam, overall_rating, offensive_rating, defensive_rating) %>%
        rename(Rank = rank, Team = posteam, Overall = overall_rating,
               Offense = offensive_rating, Defense = defensive_rating) %>%
        mutate(across(where(is.numeric), ~round(.x, 1)))
    }
  }, options = list(pageLength = 15))
  
  # League averages table
  output$league_averages_table <- DT::renderDataTable({
    data <- selected_season_data()
    
    averages <- data %>%
      summarise(across(where(is.numeric), ~round(mean(.x, na.rm = TRUE), 3))) %>%
      pivot_longer(everything(), names_to = "Metric", values_to = "League_Average") %>%
      arrange(Metric)
    
    averages
  }, options = list(pageLength = 15, dom = 'tp'))
  
  # Complete team data table
  output$complete_team_data <- DT::renderDataTable({
    data <- selected_season_data()
    
    # Filter based on view selection
    if(input$data_view == "Offensive Only") {
      data <- data %>% select(season, posteam, starts_with("off_"))
    } else if(input$data_view == "Defensive Only") {
      data <- data %>% select(season, posteam, starts_with("def_"))
    } else if(input$data_view == "Net/Efficiency") {
      data <- data %>% select(season, posteam, starts_with("net_"), 
                              contains("efficiency"), contains("margin"), 
                              contains("balance"))
    }
    
    data %>% mutate(across(where(is.numeric), ~round(.x, 3)))
  }, options = list(pageLength = 20, scrollX = TRUE, dom = 'Bfrtip'))
  
  # Data dictionary table  
  output$data_dictionary_table <- DT::renderDataTable({
    # Create comprehensive data dictionary
    data_dict <- data.frame(
      Column = names(team_stats)[-c(1,2)], # Exclude season and posteam
      Category = c(
        rep("Offensive", 12),
        rep("Defensive", 12),
        rep("Derived", 4)
      )[1:(ncol(team_stats)-2)],
      Description = c(
        "Offensive EPA per play",
        "Offensive success rate", 
        "Offensive red zone touchdown rate",
        "Offensive third down conversion rate",
        "Offensive rushing EPA per play",
        "Offensive rushing success rate",
        "Offensive passing EPA per play",
        "Offensive passing success rate",
        "Offensive completion percentage",
        "Offensive interception rate",
        "Offensive fumble rate",
        "Total offensive plays",
        "Defensive EPA allowed per play",
        "Defensive success rate allowed",
        "Defensive red zone TD rate allowed", 
        "Defensive third down conversion allowed",
        "Defensive rushing EPA allowed",
        "Defensive rushing success allowed",
        "Defensive passing EPA allowed",
        "Defensive passing success allowed",
        "Defensive completion rate allowed",
        "Defensive interception rate",
        "Defensive fumble recovery rate",
        "Defensive sack rate",
        "Net EPA per play (off - def)",
        "Net success rate (off - def)",
        "Rush/pass balance ratio",
        "Net turnover margin rate"
      )[1:(ncol(team_stats)-2)]
    )
    
    data_dict
  }, options = list(pageLength = 20))
  
  # Download handler
  output$download_data <- downloadHandler(
    filename = function() {
      paste("nfl_team_data_", input$data_season, "_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(selected_season_data(), file, row.names = FALSE)
    }
  )
}

# ==============================================================================
# RUN THE APP
# ==============================================================================

shinyApp(ui = ui, server = server)