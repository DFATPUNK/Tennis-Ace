## This project is part of the MACHINE LEARNING/AI ENGINEER career path
## Syllabus: https://www.codecademy.com/learn/paths/machine-learning-engineer

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('./tennis_stats.csv')

print(df.head(5))
print(df.columns)
# [['FirstServe', 'FirstServePointsWon',
#       'FirstServeReturnPointsWon', 'SecondServePointsWon',
#       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
#      'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
#       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
#       'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
#       'TotalPointsWon', 'TotalServicePointsWon']]

## Uncomment sections below to run tests and visualize scatter plots

## perform single feature linear regressions here:

# x = df[['BreakPointsOpportunities']]
# y = df[['Winnings']]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# mlr = LinearRegression()
# model = mlr.fit(x_train, y_train)
# y_predict = mlr.predict(x_test)

# Scores (all columns vs wins)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))
# 0.8122102844779242
# 0.8046675667786669

# perform exploratory analysis here:

# plt.scatter(y_test, y_predict, alpha=0.4)
# plt.xlabel('Winnings')
# plt.ylabel('Predicted Winnings')
# plt.show()

## perform two feature linear regressions here:

# x_2 = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
# y_2 = df[['Wins']]

# IMPORTANT LEARNING HERE: Variables order matters! x & _train first, y & _test last (error thrown otherwise)
# x_2_train, x_2_test, y_2_train, y_2_test = train_test_split(x_2, y_2, train_size=0.8, test_size=0.2)

# lm_2 = LinearRegression()
# model_2 = lm_2.fit(x_2_train, y_2_train)
# y_2_predict = lm_2.predict(x_2_test)

# Scores (all columns vs wins)
# print(model.score(x_2_train, y_2_train))
# print(model.score(x_2_test, y_2_test))
# 0.8459638858122929
# 0.8901439435810833

# plt.scatter(y_2_test, y_2_predict, alpha=0.4)
# plt.xlabel('Winnings')
# plt.ylabel('Predicted Winnings')
# plt.show()

## perform multiple feature linear regressions here:

features = df[['FirstServe', 'FirstServePointsWon',
       'FirstServeReturnPointsWon', 'SecondServePointsWon',
       'SecondServeReturnPointsWon', 'Aces', 'BreakPointsConverted',
      'BreakPointsFaced', 'BreakPointsOpportunities', 'BreakPointsSaved',
       'DoubleFaults', 'ReturnGamesPlayed', 'ReturnGamesWon',
       'ReturnPointsWon', 'ServiceGamesPlayed', 'ServiceGamesWon',
       'TotalPointsWon', 'TotalServicePointsWon']]
outcome = df[['Winnings']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size=0.8, test_size=0.2)

mlr = LinearRegression()
model = mlr.fit(features_train, outcome_train)
outcome_predict = mlr.predict(features_test)

# Coefficients
# print(model.coef_)
# [[ 2.77816002e+00  1.21392683e+01 -3.98260790e-01  6.31279736e+00
#  -4.19120976e+00  8.05108875e-03  1.47326650e-01 -3.72373137e-02
#  6.45039704e-02  2.40764145e+00 -8.21074069e-03 -5.65070783e-02
#   4.52781632e+00  3.22713391e+00  8.17434208e-02 -1.75420265e+00
# -2.00467450e+01 -8.20713413e+00]]

# Scores (all columns vs wins)
print(model.score(features_train, outcome_train))
print(model.score(features_test, outcome_test))
# 0.8772525886323974
# 0.9322143653696723

# perform exploratory analysis here:

plt.scatter(outcome_test, outcome_predict, alpha=0.4)
plt.xlabel('Winnings')
plt.ylabel('Predicted Winnings')
plt.show()