library(ggplot2)
library(caret)
library(earth)
library(dplyr)
library(tidyverse)
library(vip)
library(pdp)


train = train %>% select(-c(X, Id))
colnames(train)
test = test %>% select(-c(X, Id))
colnames(test)

# Fitting in the model 
fit = earth(ylogSalePrice~., data = train, degree = 2)

# The model summary
summary(fit)

# looking at the interactions

fit2 = earth(ylogSalePrice ~., 
             data = train, 
             degree = 2)

# Checking out the first 10 coef terms
summary(fit2) %>% .$coefficients %>% head(10)


# Tuning

# creating a tuning grid

hyper_grid = expand.grid(
  degree = 1:3, 
  nprune = seq(2,100, length.out = 10) %>% floor()
)
head(hyper_grid)

# creating a cross validated model

set.seed(123)
cv_fit = train(x = subset(train, select = -ylogSalePrice), 
               y = train$ylogSalePrice, 
               method = 'earth', 
               metric = 'RMSE', 
               trControl = trainControl(method = 'cv', number = 10), 
               tuneGrid = hyper_grid)
# Viewing the results 

cv_fit$bestTune

ggplot(cv_fit)

# variable importance plots
p1 <- vip(cv_fit, num_features = 20, bar = FALSE, value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_fit, num_features = 20, bar = FALSE, value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)

#  we see that TotalSF and Year_Built are the two most influential variables;
#however, variable importance does not tell us how our model is treating the non-linear patterns for each feature.
#Also, if we look at the interaction terms our model retained, we see interactions between different hinge functions 
#for TotalSF and Year_Built.
# extract coefficients, convert to tidy data frame, and
# filter for interaction terms
cv_fit$finalModel %>%
  coef() %>%  
  broom::tidy() %>%  
  filter(stringr::str_detect(names, "\\*")) 

#To better understand the relationship between these features and Sale_Price, 
#we can create partial dependence plots (PDPs) for each feature individually and also together.
#The individual PDPs illustrate that our model found that one knot in each feature provides the best fit.
# Construct partial dependence plots
p1 <- partial(cv_fit, pred.var = "TotalSF", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(cv_fit, pred.var = "YearBuilt", grid.resolution = 10) %>% 
  autoplot()
p3 <- partial(cv_fit, pred.var = c("TotalSF", "YearBuilt"), 
              grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
# Partial dependence plots to understand the relationship between Sale_Price, TotalSF and the Year_Built,  features. The PDPs tell
#us that as TotalSF increases and for newer homes, Sale_Price increases dramatically.
