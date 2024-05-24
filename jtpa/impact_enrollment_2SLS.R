# script to replicate the impact per enrollment estimates in the paper

# here using cleaned data from python script - which gets ride of entries with corrupt binary vars

# load data
library(tidyverse)
library(AER)
library(estimatr)
library(broom)
library(modelsummary)

data <- read_csv("jtpa_doubleclean.csv")

# get vector of features (with sex for joint model and without for sexist models)
features <- c('afdc', 'sex', 'married', 'pbhous', 'hsorged', 'black', 'hispanic', 'wkless13','age', 'prevearn')
features_sexist <- c('afdc', 'married', 'pbhous', 'hsorged', 'black', 'hispanic', 'wkless13','age', 'prevearn')

# create a string that contains all the feature names separated by "+"
features_str <- paste(features, collapse = " + ")
features_sexist_str <- paste(features_sexist, collapse = " + ")

# first run OLS with HC0 standard errors
ols_joint <- lm_robust(as.formula(paste("earnings ~ assignmt +", features_str)), data = data, se_type = "HC0")
ols_m <- lm_robust(as.formula(paste("earnings ~ assignmt +", features_sexist_str)), data = data |> filter(sex==1), se_type = "HC0")
ols_f <- lm_robust(as.formula(paste("earnings ~ assignmt +", features_sexist_str)), data = data |> filter(sex==0), se_type = "HC0")
# these are the same as the statsmodels results in Python !


# run 2SLS on earnings ~ enrollment + features and use assignment as instrument
iv_m <- AER::ivreg(as.formula(paste("earnings ~ training +", features_sexist_str, "| assignmt +", features_sexist_str)), data = data |> filter(sex==1))
iv_m |> tidy()

iv_f<- AER::ivreg(as.formula(paste("earnings ~ training +", features_sexist_str, "| assignmt +", features_sexist_str)), data = data |> filter(sex==0))
iv_f |> tidy()

# put all the regression results into a dataframe
reg_list  <- list("OLS Joint" = ols_joint, "OLS Male" = ols_m, "OLS Female" = ols_f, "IV Male" = iv_m, "IV Female" = iv_f)

# omit all but assignmt and training
coef_omit_list <- paste(c("(Intercept)",features), collapse = "|")

# Create the model summary
results <- modelsummary(reg_list, stars = FALSE, gof_omit = 'Log.Lik|R2 Adj.|AIC|BIC|F', fmt = 2, coef_omit = coef_omit_list)
results

# Save to latex
modelsummary(reg_list, output = "regression_table_R.tex", stars = FALSE, gof_omit = 'Log.Lik|R2 Adj.|AIC|BIC|F', fmt = "2f", coef_omit = coef_omit_list)
