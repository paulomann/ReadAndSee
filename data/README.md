# Data folder structure

The data is not present in the project due to ethical constraint. Since we are under the approval of the ethics committee
of Universidade Federal Fluminense (UFF), the Instagram data is entirely restricted to the research team. However, here
we show how we use the folder structure to complete the project.

```
data
│   README.md
│
|───raw                                <- The original, immutable data dump (questionnaire data).
|   |   questionnaire.csv
|   |
└───external                           <- Data from third party sources (Instagram posts)
│   │   [user_1, user_2, ... , user_n] <- A list of folders containing Instagram posts data for each user
│   │
|───interim                            <- Intermediate data that has been transformed
│   │   instagram.csv                  <- Data from questionnaire with only "valid" instagram usernames*
│   │   twitter.csv                    <- Data from questionnaire with only "valid" twitter usernames*
|   |   all_participants.csv           <- All data from all users
│   
└───processed                          <- The final, canonical data sets for modeling.
|   │   instagram.csv                  <- Users in which their profile has at least 1 public post available
|   │   TODO
|   |
```

\* "Valid" usernames do not contain *spaces*, are not *digits*, and has at least *length 1*
