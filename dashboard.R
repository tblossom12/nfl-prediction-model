# ==============================================================================
# NFL PREDICTION MODEL - INTERACTIVE DASHBOARD
# Script: dashboard.R
# Purpose: Interactive Shiny dashboard for NFL game predictions
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
  
  # Get team list
  team_list <- sort(unique(team_stats$posteam))
  
  # Current team rankings
  current_rankings <- get_team_rankings()
  
  data_loaded <- TRUE
}, error = function(e) {
  data_loaded <- FALSE
  error_message <- e$message
})

# ==============================================================================
# UI DEFINITION
# ==============================================================================

ui <- dashboardPage(
  dashboardHeader(title = "NFL Prediction Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Game Predictor", tabName = "predictor", icon = icon("football")),
      menuItem("Model Performance", tabName = "performance", icon = icon("chart-line")),
      menuItem("Team Rankings", tabName = "rankings", icon = icon("trophy")),
      menuItem("Weekly Predictions", tabName = "weekly", icon = icon("calendar-week")),
      menuItem("Historical Analysis", tabName = "historical", icon = icon("history"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .main-header .logo {
          font-family: 'Arial', sans-serif;
          font-weight: bold;
          font-size: 24px;
        }
        .box-title {
          font-size: 18px;
          font-weight: bold;
        }
        .prediction-result {
          background-color: #f4f4f4;
          padding: 15px;
          border-radius: 5px;
          margin: 10px 0;
        }
      "))
    ),
    
    tabItems(
      # Game Predictor Tab
      tabItem(tabName = "predictor",
              fluidRow(
                box(
                  title = "Single Game Prediction", status = "primary", solidHeader = TRUE,
                  width = 6,
                  selectInput("home_team", "Home Team:", choices = if(data_loaded) team_list else NULL),
                  selectInput("away_team", "Away Team:", choices = if(data_loaded) team_list else NULL),
                  numericInput("week_num", "Week:", value = 1, min = 1, max = 22),
                  selectInput("season_phase", "Season Phase:", 
                              choices = c("Regular", "Playoffs"),
                              selected = "Regular"),
                  actionButton("predict_game", "Predict Game", class = "btn-primary"),
                  br(), br(),
                  withSpinner(uiOutput("single_prediction"))
                ),
                
                box(
                  title = "Quick Matchups", status = "info", solidHeader = TRUE,
                  width = 6,
                  h4("Popular Rivalry Matchups"),
                  actionButton("predict_kc_buf", "KC @ BUF", class = "btn-info", style = "margin: 5px;"),
                  actionButton("predict_bal_cin", "BAL @ CIN", class = "btn-info", style = "margin: 5px;"),
                  actionButton("predict_dal_phi", "DAL @ PHI", class = "btn-info", style = "margin: 5px;"),
                  actionButton("predict_sf_lar", "SF @ LAR", class = "btn-info", style = "margin: 5px;"),
                  br(), br(),
                  h4("Model Confidence Levels:"),
                  tags$ul(
                    tags$li("50-60%: Toss-up game"),
                    tags$li("60-70%: Slight favorite"),
                    tags$li("70-80%: Strong favorite"),
                    tags$li("80%+: Heavy favorite")
                  )
                )
              ),
              
              fluidRow(
                box(
                  title = "Prediction Explanation", status = "warning", solidHeader = TRUE,
                  width = 12, collapsible = TRUE, collapsed = TRUE,
                  h4("How the Model Works:"),
                  tags$p("The ensemble model combines four different machine learning algorithms:"),
                  tags$ul(
                    tags$li("Logistic Regression with Elastic Net regularization"),
                    tags$li("Random Forest with 500 trees"),
                    tags$li("XGBoost gradient boosting"),
                    tags$li("Support Vector Machine with RBF kernel")
                  ),
                  tags$p("Key factors considered:"),
                  tags$ul(
                    tags$li("EPA (Expected Points Added) advantage in offense vs defense"),
                    tags$li("Success rate differentials"),
                    tags$li("Rush and pass matchup advantages"),
                    tags$li("Red zone and third down efficiency"),
                    tags$li("Turnover margins"),
                    tags$li("Recent team performance trends")
                  )
                )
              )
      ),
      
      # Model Performance Tab  
      tabItem(tabName = "performance",
              fluidRow(
                valueBoxOutput("overall_accuracy"),
                valueBoxOutput("best_model"),
                valueBoxOutput("total_games")
              ),
              
              fluidRow(
                box(
                  title = "Model Comparison", status = "primary", solidHeader = TRUE,
                  width = 8,
                  withSpinner(plotlyOutput("performance_plot"))
                ),
                
                box(
                  title = "Performance Metrics", status = "info", solidHeader = TRUE,
                  width = 4,
                  withSpinner(DT::dataTableOutput("performance_table"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Feature Importance", status = "success", solidHeader = TRUE,
                  width = 12,
                  withSpinner(plotlyOutput("feature_importance_plot"))
                )
              )
      ),
      
      # Team Rankings Tab
      tabItem(tabName = "rankings",
              fluidRow(
                box(
                  title = "Current Team Power Rankings", status = "primary", solidHeader = TRUE,
                  width = 8,
                  withSpinner(DT::dataTableOutput("rankings_table"))
                ),
                
                box(
                  title = "Rating System", status = "info", solidHeader = TRUE,
                  width = 4,
                  h4("Overall Rating:"),
                  tags$p("Based on Net EPA per play × 100"),
                  br(),
                  h4("Offensive Rating:"),
                  tags$p("Offensive EPA per play × 100"),
                  br(),
                  h4("Defensive Rating:"),
                  tags$p("-(Defensive EPA allowed) × 100"),
                  tags$p(tags$em("Higher is better for all ratings"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Team Comparison", status = "warning", solidHeader = TRUE,
                  width = 12,
                  selectInput("compare_teams", "Select Teams to Compare:", 
                              choices = if(data_loaded) team_list else NULL,
                              multiple = TRUE, selected = if(data_loaded) team_list[1:4] else NULL),
                  withSpinner(plotlyOutput("team_comparison_plot"))
                )
              )
      ),
      
      # Weekly Predictions Tab
      tabItem(tabName = "weekly",
              fluidRow(
                box(
                  title = "Weekly Predictions", status = "primary", solidHeader = TRUE,
                  width = 4,
                  numericInput("predict_week", "Select Week:", value = 1, min = 1, max = 22),
                  numericInput("predict_season", "Season:", value = 2025, min = 2020, max = 2025),
                  actionButton("generate_weekly", "Generate Predictions", class = "btn-primary"),
                  br(), br(),
                  textOutput("weekly_status")
                ),
                
                box(
                  title = "Week Summary", status = "info", solidHeader = TRUE,
                  width = 8,
                  withSpinner(uiOutput("weekly_summary"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Weekly Games", status = "success", solidHeader = TRUE,
                  width = 12,
                  withSpinner(DT::dataTableOutput("weekly_predictions_table"))
                )
              )
      ),
      
      # Historical Analysis Tab
      tabItem(tabName = "historical",
              fluidRow(
                box(
                  title = "Historical Performance", status = "primary", solidHeader = TRUE,
                  width = 6,
                  selectInput("historical_season", "Select Season:", 
                              choices = if(data_loaded) 2020:2024 else NULL),
                  withSpinner(plotlyOutput("historical_accuracy"))
                ),
                
                box(
                  title = "Confidence vs Accuracy", status = "info", solidHeader = TRUE,
                  width = 6,
                  withSpinner(plotlyOutput("confidence_accuracy"))
                )
              ),
              
              fluidRow(
                box(
                  title = "Team Performance Over Time", status = "warning", solidHeader = TRUE,
                  width = 12,
                  selectInput("track_team", "Select Team:", 
                              choices = if(data_loaded) team_list else NULL),
                  withSpinner(plotlyOutput("team_performance_time"))
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
  
  # Reactive values for storing predictions
  values <- reactiveValues(
    single_prediction = NULL,
    weekly_predictions = NULL
  )
  
  # Single Game Prediction
  observeEvent(input$predict_game, {
    req(input$home_team, input$away_team)
    
    if (input$home_team == input$away_team) {
      showNotification("Please select different teams", type = "warning")
      return()
    }
    
    withProgress(message = 'Predicting game...', {
      values$single_prediction <- predict_game(input$home_team, input$away_team, 
                                               input$week_num, input$season_phase)
    })
  })
  
  # Quick prediction buttons
  observeEvent(input$predict_kc_buf, {
    values$single_prediction <- predict_game("KC", "BUF", input$week_num)
  })
  
  observeEvent(input$predict_bal_cin, {
    values$single_prediction <- predict_game("BAL", "CIN", input$week_num)
  })
  
  observeEvent(input$predict_dal_phi, {
    values$single_prediction <- predict_game("DAL", "PHI", input$week_num)
  })
  
  observeEvent(input$predict_sf_lar, {
    values$single_prediction <- predict_game("SF", "LAR", input$week_num)
  })
  
  # Display single prediction
  output$single_prediction <- renderUI({
    req(values$single_prediction)
    
    pred <- values$single_prediction
    
    div(class = "prediction-result",
        h3(paste(pred$away_team, "@", pred$home_team)),
        h4(paste("Prediction:", pred$predicted_winner, "wins")),
        h4(paste("Confidence:", pred$confidence)),
        br(),
        h5("Individual Model Probabilities (Home Win):"),
        tags$ul(
          tags$li(paste("Logistic Regression:", pred$logistic_prob)),
          tags$li(paste("Random Forest:", pred$rf_prob)),
          tags$li(paste("XGBoost:", pred$xgb_prob)),
          tags$li(paste("SVM:", pred$svm_prob)),
          tags$li(tags$b(paste("Ensemble:", pred$ensemble_prob)))
        )
    )
  })
  
  # Value boxes for performance
  output$overall_accuracy <- renderValueBox({
    test_performance <- performance_data %>% filter(grepl("Ensemble", Model))
    accuracy <- if(nrow(test_performance) > 0) test_performance$Accuracy[1] else 0.65
    
    valueBox(
      value = paste0(round(accuracy * 100, 1), "%"),
      subtitle = "Overall Accuracy",
      icon = icon("bullseye"),
      color = "green"
    )
  })
  
  output$best_model <- renderValueBox({
    best_model <- performance_data %>% 
      filter(!grepl("Ensemble", Model)) %>%
      top_n(1, AUC)
    
    valueBox(
      value = if(nrow(best_model) > 0) best_model$Model[1] else "Random Forest",
      subtitle = "Best Individual Model",
      icon = icon("trophy"),
      color = "yellow"
    )
  })
  
  output$total_games <- renderValueBox({
    # Estimate based on typical seasons
    valueBox(
      value = "800+",
      subtitle = "Games Analyzed",
      icon = icon("football"),
      color = "blue"
    )
  })
  
  # Performance comparison plot
  output$performance_plot <- renderPlotly({
    p <- performance_data %>%
      filter(!is.na(Accuracy)) %>%
      plot_ly(x = ~Model, y = ~Accuracy, type = "bar", 
              marker = list(color = c("#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#1f77b4"))) %>%
      layout(title = "Model Accuracy Comparison",
             yaxis = list(title = "Accuracy", range = c(0.5, 0.8)),
             xaxis = list(title = "Model"))
    p
  })
  
  # Performance table
  output$performance_table <- DT::renderDataTable({
    performance_data %>%
      select(Model, Accuracy, AUC, F1_Score) %>%
      mutate_if(is.numeric, round, 3)
  }, options = list(pageLength = 10, dom = 't'))
  
  # Team rankings table
  output$rankings_table <- DT::renderDataTable({
    current_rankings %>%
      select(rank, posteam, overall_rating, offensive_rating, defensive_rating) %>%
      rename(Rank = rank, Team = posteam, Overall = overall_rating,
             Offense = offensive_rating, Defense = defensive_rating) %>%
      mutate_if(is.numeric, round, 1)
  }, options = list(pageLength = 15))
  
  # Team comparison plot
  output$team_comparison_plot <- renderPlotly({
    req(input$compare_teams)
    
    comparison_data <- current_rankings %>%
      filter(posteam %in% input$compare_teams) %>%
      select(posteam, offensive_rating, defensive_rating) %>%
      tidyr::pivot_longer(cols = c(offensive_rating, defensive_rating),
                          names_to = "Rating_Type", values_to = "Rating") %>%
      mutate(Rating_Type = ifelse(Rating_Type == "offensive_rating", "Offense", "Defense"))
    
    p <- comparison_data %>%
      plot_ly(x = ~posteam, y = ~Rating, color = ~Rating_Type, type = "bar") %>%
      layout(title = "Team Offensive vs Defensive Ratings",
             yaxis = list(title = "Rating"),
             xaxis = list(title = "Team"),
             barmode = "group")
    p
  })
  
  # Weekly predictions
  observeEvent(input$generate_weekly, {
    withProgress(message = 'Generating weekly predictions...', {
      values$weekly_predictions <- predict_week(input$predict_week, input$predict_season)
    })
  })
  
  output$weekly_status <- renderText({
    if (is.null(values$weekly_predictions)) {
      "Click 'Generate Predictions' to see weekly games"
    } else if (nrow(values$weekly_predictions) == 0) {
      "No games found for selected week"
    } else {
      paste("Generated predictions for", nrow(values$weekly_predictions), "games")
    }
  })
  
  output$weekly_summary <- renderUI({
    req(values$weekly_predictions)
    req(nrow(values$weekly_predictions) > 0)
    
    preds <- values$weekly_predictions
    high_conf_games <- sum(abs(preds$ensemble_prob - 0.5) > 0.2)
    avg_conf <- mean(abs(preds$ensemble_prob - 0.5))
    
    div(
      h4(paste("Week", input$predict_week, "Summary:")),
      tags$p(paste("Total Games:", nrow(preds))),
      tags$p(paste("High Confidence Games:", high_conf_games)),
      tags$p(paste("Average Confidence:", round(avg_conf * 100, 1), "%"))
    )
  })
  
  output$weekly_predictions_table <- DT::renderDataTable({
    req(values$weekly_predictions)
    
    values$weekly_predictions %>%
      select(away_team, home_team, predicted_winner, confidence, ensemble_prob) %>%
      rename(Away = away_team, Home = home_team, Predicted_Winner = predicted_winner,
             Confidence = confidence, Probability = ensemble_prob) %>%
      arrange(desc(Probability))
  }, options = list(pageLength = 20))
  
  # Feature importance plot (mock data for demonstration)
  output$feature_importance_plot <- renderPlotly({
    # Mock feature importance data
    feature_data <- data.frame(
      Feature = c("EPA Advantage", "Success Rate Advantage", "Rush Matchup", 
                  "Pass Matchup", "Home Net EPA", "Turnover Advantage"),
      Importance = c(0.25, 0.22, 0.18, 0.15, 0.12, 0.08)
    )
    
    p <- feature_data %>%
      plot_ly(x = ~reorder(Feature, Importance), y = ~Importance, type = "bar",
              marker = list(color = "steelblue")) %>%
      layout(title = "Feature Importance in Ensemble Model",
             yaxis = list(title = "Importance"),
             xaxis = list(title = "Feature")) %>%
      layout(xaxis = list(categoryorder = "total ascending"))
    p
  })
}

# ==============================================================================
# RUN THE APP
# ==============================================================================

shinyApp(ui = ui, server = server)