# CaseStudy Machine learning Neural Network - Churn identification

AB Gaming is a digital gaming platform that offers a wide variety of games available under monthly subscriptions. There are 3 different plans, labeled SMALL, MEDIUM and LARGE, and can be paid in USD or EUR. 

You are provided with 2 datasets:
- sales.csv: contains the sales of clients acquired since 2019-01-01. Data was extracted on 2020-12-31. This dataset includes a unique identifier for each user (account_id).
- user_activity.csv: this dataset contains the following user characteristics as well as their unique identifier (same as in sales.csv):

  - gender: gender of user reported in their profile.
  - age: age in completed years of the user at the beginning of their subscription.
  - type: device type the user has installed the gaming platform.
  - genre1: most played game genre by the user.
  - genre2: second most played game genre by the user.
  - hours: mean number of hours played by the user weekly.
  - games: median number of different games played by the user weekly


Create a ML model to predict subscribers’ churn.

We define churn as the users that stop their subscription before their 6th renewal. Hence, a user that has less than 7 orders of payment is considered a churner. For this model we will be using activity for the first 3 months, so those users that have made only 1 or 2 payments should not be included in the model.

To create the model, extract relevant data (such as churner) from the sales.csv dataset and join it with the user_activity.csv dataset.

In the results of your report remember to answer these questions:

What is the accuracy of your model? (consider also sensibility, specificity, PPV and NPV)
Do you consider it a good predictive model?
In the conclusion/discussion of your report make sure to mention any limitations as well as ways to enhance future iterations of the creation of the model.
