# ==============================================================================
# LAUNCH NFL PREDICTION DASHBOARD
# Script: launch_dashboard.R
# Purpose: Simple script to launch the Shiny dashboard
# ==============================================================================

# Install required packages if not already installed
required_packages <- c("shiny", "shinydashboard", "DT", "plotly", 
                       "shinycssloaders", "here", "dplyr", "readr")

missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(missing_packages) > 0) {
  install.packages(missing_packages)
}

# Load the dashboard
cat("Launching NFL Prediction Dashboard...\n")
cat("The dashboard will open in your default web browser.\n")
cat("To stop the dashboard, press Ctrl+C or Esc in the R console.\n\n")

# Set working directory to project root if using here()
library(here)
setwd(here())

# Run the dashboard
shiny::runApp("dashboard.R", launch.browser = TRUE, port = 3838)