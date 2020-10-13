---
title: "IST 707 - Data Analytics: Final Project"
author: "Jeremy Tobias"
date: "11/1/2019"
output: 
  html_document:
    code_folding: "hide"
    toc: true
    toc_float: true
    fig_width: 12
    fig_height: 6
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
```

# __Data Science Bowl 2019: Uncover the factors to help measure how young children learn__

# __Introduction__

From [DSB 2019 Overview](https://www.kaggle.com/c/data-science-bowl-2019/overview)

>Uncover new insights in early childhood education and how media can support learning outcomes. Participate in our fifth annual Data Science Bowl, presented by Booz Allen Hamilton and Kaggle.

>PBS KIDS, a trusted name in early childhood education for decades, aims to gain insights into how media can help children learn important skills for success in school and life. In this challenge, you’ll use anonymous gameplay data, including knowledge of videos watched and games played, from the PBS KIDS Measure Up! app, a game-based learning tool developed as a part of the CPB-PBS Ready To Learn Initiative with funding from the U.S. Department of Education. Competitors will be challenged to predict scores on in-game assessments and create an algorithm that will lead to better-designed games and improved learning outcomes. Your solutions will aid in discovering important relationships between engagement with high-quality educational media and learning processes...

>...In the PBS KIDS Measure Up! app, children ages 3 to 5 learn early STEM concepts focused on length, width, capacity, and weight while going on an adventure through Treetop City, Magma Peak, and Crystal Caves. Joined by their favorite PBS KIDS characters, children can also collect rewards and unlock digital toys as they play...

The goal is this project will be to try to predict not only whether or not an individual will be able to complete an assessment, but to also predict the accuracy group in which that individual might fall.

Accuracy groups are broken into 4 categories defined as:

* 3: the assessment was solved on the first attempt
* 2: the assessment was solved on the second attempt
* 1: the assessment was solved after 3 or more attempts
* 0: the assessment was never solved

```{r}
# Libraries
library(tidyverse)
library(doParallel)
library(caret)
library(FSelector)
library(mlbench)
library(rmarkdown)

# House keeping

# set the number of CPU cores to use
cl <- makePSOCKcluster(6)
registerDoParallel(cl)

# adjust default theme settings
theme_set(theme_minimal() +
            theme(axis.title.x = element_text(size = 14, hjust = 1),
                  axis.title.y = element_text(size = 14),
                  axis.text.x = element_text(size = 9),
                  axis.text.y = element_text(size = 9),
                  panel.grid.major = element_line(linetype = 2),
                  panel.grid.minor = element_line(linetype = 2),
                  plot.title = element_text(size = 18, color = "grey29", face = "bold"),
                  plot.subtitle = element_text(size = 16, colour = "dimgrey"),
                  legend.position = "bottom"))
```

## __Load Data__

The DSB:2019 provides 4 main files (train.csv, test.csv, train_labels.csv, specs.csv) and a sample submission file (sample_submission.csv). At the current writing of this project we will not be utilizing specs.csv. However, cursory analysis of the specs file suggests that there might be relevant and important data that could aid us in future prediction efforts.

```{r}
pbs_train <- read_csv("train.csv")

pbs_test <- read_csv("test.csv") 

labels <- read_csv("train_labels.csv")

specs <- read_csv("specs.csv")
```

# __Exploratory Data Analysis__


## __Default observations and features__

Below we can see the raw dimensions for each file.

```{r}
all_dims <- data.frame("train.csv" = dim(pbs_train), "test.csv" = dim(pbs_test), "label.csv" = dim(labels), "specs.csv" = dim(specs))

row.names(all_dims) <- c("observations", "features")

paged_table(all_dims)
```

## __Data Files__

### __train.csv & test.csv__
Contain the gameplay events below:

* __event_id__ - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
* __game_session__ - Randomly generated unique identifier grouping events within a single game or video play session.
* __timestamp__ - Client-generated datetime
* __event_data__ - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.
* __installation_id__ - Randomly generated unique identifier grouping game sessions within a single installed application instance.
* __event_count__ - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
* __event_code__ - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.
* __game_time__ - Time in milliseconds since the start of the game session. Extracted from event_data.
* __title__ - Title of the game or video.
* __type__ - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
* __world__ - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

```{r}
paged_table(pbs_train, options = list(max.print = 10))
```

### __train_labels.csv__
Demonstrates how to compute the ground truth for the assessments in the training set.

* __game_session__ - Randomly generated unique identifier grouping events within a single game or video play session.
* __installation_id__ - Randomly generated unique identifier grouping game sessions within a single installed application instance.
* __title__ - Title of the game or video.
* __accuracy_group__ - defined above, and determined by accuracy

The remainder of the features do not have formal definitions given, but we can make some assumptions based of the data given.

* __num_correct__ - Number of correct assessment attempts for game session
* __num_incorrect__ - Number of incorrect assessment attempts for game session
* __accuracy__ - num_correct divided by total attempts for game session

```{r}
paged_table(labels, options = list(max.print = 10))
```

### __specs.csv__
This file gives the specification of the various event types.

* __event_id__ - Global unique identifier for the event type. Joins to event_id column in events table.
* __info__ - Description of the event.
* __args__ - JSON formatted string of event arguments. Each argument contains:
  * __name__ - Argument name.
  * __type__ - Type of the argument (string, int, number, object, array).
  * __info__ - Description of the argument.

```{r}
paged_table(specs, options = list(max.print = 10))
```

## __Uniques__

From our preliminary analysis we see that the training set has over 11MM observations, an incredibly large number observations.

By digging a little deeper we uncover that there are 17,000 unique IDs over the whole dataset.

Digging a little deeper we can see what the breakdown is over each media type. From the information given, we know that there are 4 media types; 'Game', 'Assessment', 'Activity', and 'Clip'. We will also compare the unique IDs over the training data and the labels data.

```{r}
total_ids <- pbs_train %>%
  summarise(total_train_ids = n_distinct(installation_id))

in_labels <- labels %>%
  summarise(total_label_ids = n_distinct(installation_id))

train_in_labels <- pbs_train %>%
  filter(installation_id %in% labels$installation_id) %>%
  summarize(train_label_ids = n_distinct(installation_id))

type_uniques <- pbs_train %>%
  group_by(type) %>%
  summarise(unique_type_ids = n_distinct(installation_id))

unique_set_ids <- data.frame(total_ids, in_labels, train_in_labels, type_uniques) %>%
  pivot_wider(names_from = type, values_from = unique_type_ids) %>%
  pivot_longer(cols = c(total_train_ids, 
                        total_label_ids,
                        train_label_ids,
                        Activity, 
                        Assessment, 
                        Clip, 
                        Game) , 
               names_to = c("category"), 
               names_pattern = "(.*)",
               values_to = "unique_ids")

unique_set_ids %>%
  ggplot(aes(reorder(category, unique_ids), unique_ids)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = unique_ids), hjust = 1.25) +
  coord_flip() +
  labs(x = "Unique ID Categories", y = "Num Unique IDs") +
  ggtitle("Breakdown of unique IDs")
```

Our ID breakdown reveals some interesting and good information. Most importantly, we see that of the 17,000 unique IDs only 4242 of them participated in taking assessments. Of those 4242 who took assessments, only 3614 of them are in the labels file.

It's important to reiterate that the goal of this project is to predict the outcomes of individuals who took assessments and the accuracy group in which they fall into. From this new knowledge we can eliminate the vast majority of unique IDs in the dataset.

Next we'll take a look to see if there are any overlapping IDs in the training and test set.

```{r}
test_ids <- pbs_test %>%
  summarise(test_ids = n_distinct(installation_id))

test_in_train <- pbs_test %>%
  filter(installation_id %in% pbs_train$installation_id) %>%
  summarise(in_train = n_distinct(installation_id))

paged_table(test_ids)
paged_table(test_in_train)
```

Upon inspection, we see that there are 1000 unique IDs in the test set. However, there is no overlap between the test and training set, and by transitive properties, also the labels set.

From here we can reduce the dataset down to only IDs that took assessments and are also in the labels. However, we'll keep all other data associated with those IDs to perform more analysis and, ultimately, some feature engineering on.

```{r}
did_assess <- pbs_train %>%
  filter((type == "Assessment" & installation_id %in% labels$installation_id)) %>%
  distinct(installation_id) %>%
  left_join(pbs_train, by = "installation_id") %>%
  filter(game_time > 0)

dim_assess <- data.frame("Performed Assessments" = dim(did_assess))

row.names(dim_assess) <- c("observations", "features")

paged_table(dim_assess)
```
After ID filtering is performed, we see we have reduced our dataset to now 7,542,661 observations. A reduction to 66.5% of the original. Still a relatively large number, but a good reduction in size overall.

Now, we can do some better EDA on our reduced dataset to hopefully extrapolate some more information.

## __Train set__

There appears to be a large amount of events for just a couple of the installation IDs.

```{r}
did_assess %>%
  count(installation_id, sort = T) %>%
  ggplot(aes(n)) +
  geom_histogram(fill = "steelblue", color = "orange") +
  ggtitle("Events per ID") +
  labs(x = "Count of events", y = "Number")
```

In the table below we can see the numeric breakdown for all events for each installation ID. We see that while the maximum number of events was 57,552, the median is only 1178.5. This meshes well with our graph in that one installation ID performed that majority of the events in our data. It is possible that this could skew our prediction efforts from the standpoint of the learning factors for one individual rather than across a broader sample.

```{r}
did_assess %>%
  count(installation_id, sort = T) %>%
  summarise(total = sum(n),
            mean = mean(n),
            median = median(n),
            max = max(n),
            min = min(n),
            standard_dev = sd(n)) %>%
  paged_table()
```

It appears that __Game__ were the most popular type with 3.8MM events and __Assessment__ were the lowest at 873787 events. This tracks well, since assessments generally follow after an individual has spent time performing games and activities to prepare for the assessments.

```{r}
did_assess %>%
  count(type) %>%
  ggplot(aes(type, n)) +
  geom_histogram(fill = "steelblue", color = "orange", stat = "identity") +
  geom_text(aes(label = n), vjust = 1.25) +
  ggtitle("Train Set - Media Type Frequency") +
  labs(x = "Type", y = "Count")
```

We see that __Scrub-A-Dub__ and __Chow Time__ were the two games where the most time was spent playing. It's possible that this could be due to the individual with the extremely high event cound, but further investigation is required.

```{r}
did_assess %>%
  filter(type == "Game") %>%
  group_by(title, world) %>%
  summarise(total_game_time = sum(game_time)) %>%
  group_by(world) %>%
  ggplot(aes(reorder(title, total_game_time), total_game_time, fill = world)) +
  geom_histogram(stat = "identity") +
  coord_flip() +
  labs(x = "Title", y = "Total Game Time (milliseconds)") +
  ggtitle("Train Set - Total Game Time Spent")
```

Here take a look at the average time spent on each game for the top 10 installation IDs per game. We see that while there were a couple installation IDs that dominated the majority of time on __Chow Time__, that the overall distrubtion of time was fairly good. This is true for __Scrub-A-Dub__ as well. While there seemed to be individuals who heavily favored one game over another, which skewed results for that game, in general there was no one installation ID that had the most time across a majority of the games. This will allow us to rule out that any one individual skewed the time played on games, and that it was likely the popularity of a given game which drove many individuals to play it more.

```{r}
did_assess %>%
  filter(type == "Game") %>%
  group_by(installation_id, title) %>%
  summarise(mean_game_time = mean(game_time)) %>%
  arrange(desc(mean_game_time)) %>%
  group_by(title) %>%
  slice(1:10) %>%
  ggplot(aes(reorder(installation_id, mean_game_time), mean_game_time, fill = title)) +
  geom_histogram(stat = "identity") +
  coord_flip() +
  facet_wrap(~title, scales = "free") +
  labs(x = "Installation ID", y = "Avg. Game Time (milliseconds)") +
  ggtitle("Train Set - Avg. Game Time by Installation ID (Top 10)") +
  theme(strip.text = element_text(size = 12, face = "bold"), legend.position = "hidden")
```

Below we see the breakdown across __Activity__, __Assessment__, and __Game__. Here we display the top 5 for each category. __Clip__ was removed, and at this time has been deemed not useful for our prediction process.

We see that for __Acitivity__, there is a sharp uptick for __Sandcastle Builder__ and __Bottle Filler__, similar to the uptick on __Game__, which we investigated previously. __Assessment__, however, has a nice semi-smooth curve.

```{r}
did_assess %>%
  group_by(title, type) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  group_by(type) %>%
  slice(1:5) %>%
  ggplot(aes(reorder(title, n), n)) +
  geom_histogram(fill = "steelblue", color = "orange", stat = "identity") +
  labs(x = "Title", y = "Count") +
  ggtitle("Train Set - Title Top 5") +
  facet_wrap(~type, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        strip.text = element_text(size = 12, face = "bold"))
```

Here we look at a normalized density graph of the three main activities. Again we notice __Game__ is the highest activity while __Assessment__ is the lowest.

```{r}
did_assess %>%
  ggplot(aes(log(game_time), fill = type)) +
  geom_density(color = "black", alpha = 0.5) +
  ggtitle("Time spent per media type density graph") +
  labs(x = "Game Time (log)", y = "Density")
```

## __Labels__

Much liks the training set, the labels have a heavily skewed event count for only a couple installation IDs.

```{r}
labels %>%
  count(installation_id, sort = T) %>%
  ggplot(aes(n)) +
  geom_histogram(fill = "steelblue", color = "orange") +
  ggtitle("Assessment Event Count") +
  labs(x = "Count of events", y = "Number")
```

We also see that __accuracy group 3__ has a significant number of occurences, followed by __accuracy group 0__, with roughly half the number as that of the former. As a reminder, __accuracy group 3__ completed an assessment on their first attempt, while __accuracy group 0__ did not complete the assessment.

```{r}
labels %>%
  count(accuracy_group) %>%
  ggplot(aes(reorder(accuracy_group, n), n)) +
  geom_bar(fill = "steelblue", color = "orange", stat = "identity") +
  coord_flip() +
  ggtitle("Accuracy Group Breakdown") +
  labs(x = "Accuracy Group", y = "Count")
```

It appears as though __Chest Sorter__ is the overall hardest assessment for individuals who took assessments.

```{r}
labels %>%
  count(accuracy_group, title) %>%
  ggplot(aes(title, n, fill = as.character(accuracy_group))) +
  geom_bar(position = "fill", color = "black", alpha = 0.5, stat = "identity") +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  ggtitle("Accuracy Group per Assessment Breakdown") +
  labs(fill = "Accuracy Group", x = "Title", y = "Accuracy")
```

## __Test Set__

From the information given from Kaggle, we know that while not every ID took an assessment in the training set, every installation ID in the test set took at least one assessment. Next we'll perform some EDA on the test set and see what we can discover.

We see that the test set looks relatively similar to the training set with __Game__ being the overall highest contributor to events and __Assessment__ being the least. The overally distribution of events relates well to the training set which should allow us to get similar results, pending the proper creation of our models.

```{r}
pbs_test %>%
  count(type) %>%
  ggplot(aes(type, n)) +
  geom_histogram(fill = "steelblue", color = "orange", stat = "identity") +
  geom_text(aes(label = n), vjust = 1.25) +
  ggtitle("Test Set Media Type Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

```{r}
pbs_test %>%
  filter(type != "Clip") %>%
  group_by(title, type) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  group_by(type) %>%
  slice(1:5) %>%
  ggplot(aes(reorder(title, n), n)) +
  geom_histogram(fill = "steelblue", color = "orange", stat = "identity") +
  labs(x = "Title", y = "Count") +
  ggtitle("Test Title Popularity") +
  facet_wrap(~ type, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 9, angle = 45, hjust = 1), 
        strip.text = element_text(size = 12, face = "bold"))
```

```{r}
pbs_test %>%
  filter(type != "Clip") %>%
  ggplot(aes(log(game_time), fill = type)) +
  geom_density(color = "black", alpha = 0.5) +
  ggtitle("Time spent per media type density graph") +
  labs(x = "Game Time (log)", y = "Density")
```

# __Feature Engineering__

## __Helper functions__

We'll create some helper functions to aid with our feature engineering.

* __scoring_func__ - According to Kaggle there is a JSON element in the __event_data__ variable called __correct__. We will isolate the __Game__ media type and search for this __correct__ element. If the individual got the particular event correct the __correct__ element will have been updated to __correct:true__. We'll use the total number of __correct__ instances and the total number of __correct:true__ instances to determine the accuracy of a particular game session.
* __game_stats_func__ - Isolates the __Game__ type and uses the __scoring_func__ to get the accuracy for a given game_session. We return the median over the mean due to the heavily skewed event count per installation ID we discovered earlier
* __time_stats_func__ - Returns the median time for each title per installation ID of a given media type

The following functions were used to replace the occurence of NA to see which NA replacement method worked best.

* __na_to_median__ - Replaces NAs with the median for a given feature
* __na_to_zero__ - Replaces NAs with zeros

```{r}
scoring_func <- function(df){
  install_ids <- df %>% 
    distinct(game_session, installation_id, .keep_all = T) %>%
    select(game_session, installation_id, timestamp, title)
    
  df %>% 
    filter(grepl('"correct"', event_data)) %>%
    mutate(correct = as.numeric(grepl('"correct":true', event_data))) %>%
    select(game_session, title, event_id, event_code, correct) %>%
    group_by(game_session) %>%
    summarise(accuracy = sum(correct) / n()) %>%
    select(game_session, accuracy) %>%
    left_join(., install_ids, by = c("game_session"))
}

game_stats_func <- function(df){ 
  df %>%
    filter(type == "Game") %>%
    scoring_func() %>%
    mutate(game_title = title) %>%
    group_by(installation_id, game_title) %>%
    summarise(median_accuracy = median(accuracy, na.rm = T))
}

time_stats_func <- function(df, media_type) {
  df %>%
    filter(type == media_type) %>%
    mutate(game_title = title) %>%
    group_by(installation_id, game_title) %>%
    summarise(median_time = median(game_time, na.rm = T))
}

na_to_median <- function(x) {
  replace(x, is.na(x), median(x, na.rm = TRUE))
}

na_to_zero <- function(x) {
  replace(x, is.na(x), 0)
}
```

## __Creating new features__

Now we will create a new dataset that contains installation IDs, the assessment(s) they've performed and the accuracy group that they fell into for that assessment. We will also include the game stats and game time stats created with the use of our helper functions.

To achieve this we'll combine both the training data and the label data to get a complete dataset we can perform training methods on.

```{r}
game_stats <- game_stats_func(did_assess)

game_time_stats <- time_stats_func(did_assess, "Game")

#act_time_stats <- time_stats_func(did_assess, "Activity")

game_total_stats <-game_stats %>%
  left_join(game_time_stats, by = c("installation_id", "game_title"))

labels_with_stats <- labels %>%
  left_join(game_total_stats, by = "installation_id")

labels_with_stats <- labels_with_stats %>%
  distinct(installation_id, title, game_title, .keep_all = T) %>%
  select(installation_id, title, game_title, accuracy_group, median_accuracy, median_time) %>%
  pivot_wider(names_from = game_title, values_from = c(median_accuracy, median_time))

labels_with_stats <- labels_with_stats %>%
  replace(TRUE, lapply(labels_with_stats, na_to_median)) %>%
  select(-c("median_accuracy_NA", "median_time_NA")) %>%
  mutate_at(vars(accuracy_group), funs(as.factor)) %>%
  rename_all(list(~make.names(.)))

paged_table(labels_with_stats, options = list(max.print = 10))
```

```{r}

```

*//TODO more feature engineering*

# __Modeling and Predicting__

## __Preprocessing__

Before we train our models, we'll do a little preprocessing.

For now we will use __medianImpute__, __center__, and __scale__ with __caret's preprocessing__ functions.

```{r}
preProc <- preProcess(labels_with_stats,
                      method = c("medianImpute", "center", "scale"))

labels_with_stats <- predict(preProc, labels_with_stats)
```

## __Feature Selection__

We'll use __gain ratio__ and the __chi squared test for independence__ as an attempt to discover what features are of the most importance when trying to predict the __accuracy group__. We'll use the top 15 features, which can be seen in the tables below.

__Gain Ratio__

```{r}
subset_df <- labels_with_stats %>%
  select(-c(installation_id, title))

pbs_gr <- gain.ratio(accuracy_group~.,
                     data = subset_df)

pbs_gr %>%
  rownames_to_column() %>%
  arrange(desc(attr_importance)) %>%
  paged_table(options = list(max.print = 15))

pbs_gr_subset <- cutoff.k(pbs_gr, 15)

pbs_gr_f <- as.simple.formula(pbs_gr_subset, "accuracy_group")
```

__Chi Squared Test for Independence__

```{r}
pbs_c2 <- chi.squared(accuracy_group~.,
                      data = subset_df)

pbs_c2 %>%
  rownames_to_column() %>%
  arrange(desc(attr_importance)) %>%
  paged_table(options = list(max.print = 15))

pbs_c2_subset <- cutoff.k(pbs_c2, 15)

pbs_c2_f <- as.simple.formula(pbs_c2_subset, "accuracy_group")
```

## __Creating the training and validation sets__

Now we can split our training set. We are going to use an 80/20 split, and also remove the installation ID and assessment titles, since it's likely these will have little bearing on our predictions.

```{r}
splitter <- createDataPartition(labels_with_stats$accuracy_group, p = 0.8, list = FALSE)

train_set <- labels_with_stats[splitter,]
validate_set <- labels_with_stats[-splitter,]

train_set <- train_set %>%
  select(-c(installation_id, title))
validate_set <- validate_set %>%
  select(-c(installation_id, title))
```

We'll quickly ensure the training and validation sets are well distributed across the two sets.

```{r}
ggplot(data = train_set, aes(accuracy_group)) +
  geom_bar(fill = "steelblue") +
  geom_bar(data = validate_set, aes(accuracy_group), fill ="orange") +
  ggtitle("Accuracy Group Distribution Across Training and Validation Set") +
  labs(x = "Accuracy Group", y = "Density")
```

## __Training models__

For training and prediction we will be using four training methods, each using the gain ratio and chi squared feature sets that were selected. The four training methods we'll be using are:

* __Random Forest__
* __k-Nearest Neighbor__
* __Support Vector Machines__
* __Gradient Boosting Machines__

Finally we'll use a voting ensemble method in attempt to get a consensus across all predictions, and hopefully get a better overall prediction set. 

### __Random Forest__

Our first model will be done using Caret's implementation of the random forest ranger kernel. The ranger kernel is a fast random forest kernel for highly dimensional data. We will use a 10 fold repeated cross validation, repeating 3 times using 500 trees, and a tune length of 10. Orginally, the model was trained using the default random forest kernel with 5000 trees. Ranger returned a marginally better accuracy using a smaller set of trees.

While 15 features is not necessarily highly dimensional, ranger may come in handy should we build out more robust features.

#### __RF Gain Ratio__

```{r}
rf_tc <- trainControl(method = "repeatedcv",
                      number = 10,
                      repeats = 3,
                      search = "random",
                      allowParallel = T)

pbs_rf_model <- train(pbs_gr_f,
                   data = train_set,
                   trControl = rf_tc,
                   tuneLength = 10,
                   num.trees = 500,
                   method = "ranger")

pbs_rf_pred <- predict(pbs_rf_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_rf_pred, reference = validate_set$accuracy_group)
```

#### __RF Chi Squared__

```{r}
pbs_c2_rf_model <- train(pbs_c2_f,
                         data = train_set,
                         trControl = rf_tc,
                         tuneLength = 10,
                         num.trees = 500,
                         method = "ranger")

pbs_c2_rf_pred <- predict(pbs_c2_rf_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_c2_rf_pred, reference = validate_set$accuracy_group)
```

From our predictions, we see that the chi squared feature set retured a slightly better prediction than gain ratio. An increase of ~2%.

### __k-Nearest Neighbor__

We'll next take a look at using KNN. We'll again use a 10 fold cross validation, this time with five repeats.

#### __KNN Gain Ratio__

```{r}
knn_tc <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       allowParallel = T)

pbs_knn_model <- train(pbs_gr_f,
                       data = train_set,
                       trControl = knn_tc,
                       method = "knn")

pbs_knn_pred <- predict(pbs_knn_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_knn_pred, reference = validate_set$accuracy_group)
```

#### __KNN Chi Squared__

```{r}
pbs_c2_knn_model <- train(pbs_c2_f,
                         data = train_set,
                         trControl = knn_tc,
                         method = "knn")

pbs_c2_knn_pred <- predict(pbs_c2_knn_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_c2_knn_pred, reference = validate_set$accuracy_group)
```

Our KNN predictions return a slightly worse accuracy than their respective RF predictions. Gain ratio for RF was 47.41% and KNN 46.83. Chi squared returned an accuracy of 49.33% for RF and 48.43% for KNN. We do see that there was a slight improvement on KNN chi squared accuracy for accuracy group 1, however.

### __Support Vector Machines__

#### __SVM Gain Ratio__

```{r}
svm_tc <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       allowParallel = T)

pbs_svm_model <- train(pbs_gr_f,
                       data = train_set,
                       trControl = svm_tc,
                       method = "svmRadial")

pbs_svm_pred <- predict(pbs_svm_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_svm_pred, reference = validate_set$accuracy_group)
```

#### __SVM Chi Squared__

```{r}
pbs_c2_svm_model <- train(pbs_c2_f,
                         data = train_set,
                         trControl = svm_tc,
                         method = "svmRadial")

pbs_c2_svm_pred <- predict(pbs_c2_svm_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_c2_svm_pred, reference = validate_set$accuracy_group)
```

The SVMs returned near 52% for both, beating out both the RF and KNN models. However, we see that the dataset was made almost entirely binary with the SVMs predicting almost solely on accuracy group 0 and 3.

### __Gradient Boosting Machines__

#### __GBM Gain Ratio__

```{r}
gbm_tc <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       allowParallel = T)

pbs_gbm_model <- train(pbs_gr_f,
                       data = train_set,
                       trControl = gbm_tc,
                       method = "gbm")

pbs_gbm_pred <- predict(pbs_gbm_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_gbm_pred, reference = validate_set$accuracy_group)
```

#### __GBM Chi Squared__

```{r}
pbs_c2_gbm_model <- train(pbs_c2_f,
                       data = train_set,
                       trControl = gbm_tc,
                       method = "gbm")

pbs_c2_gbm_pred <- predict(pbs_c2_gbm_model, validate_set %>% select(-accuracy_group))

caret::confusionMatrix(pbs_c2_gbm_pred, reference = validate_set$accuracy_group)
```

GBM scored relatively close to the SVMs at ~52% for both gain ratio and chi squared. We also see that although GBM made more predictions on accuracy group 1 and 2, it too predicted almost exclusively on accuracy groupo 0 and 3.

A more extensive look into how feature creation and selection is performed would probably give us a better idea of how to mitigate the data being segregated in this way.

### __Ensemble Method__

Lastly, we'll use a voting ensemble method to see if we can get a better consensus on accuracy groups across all our trained models.

```{r}
all_predicts <- data.frame(KNN = as.numeric(pbs_knn_pred) - 1,
                           KNN_C2 = as.numeric(pbs_c2_knn_pred) - 1,
                          SVM = as.numeric(pbs_svm_pred) - 1,
                          SVM_C2 = as.numeric(pbs_c2_svm_pred) - 1,
                          RF = as.numeric(pbs_rf_pred) - 1,
                          RF_c2 = as.numeric(pbs_c2_rf_pred) - 1,
                          GBM = as.numeric(pbs_gbm_pred) - 1,
                          GBM_c2 = as.numeric(pbs_c2_gbm_pred) - 1)

voting_booth <- function(row) {
   ll <- data.frame(table(unlist(row)))
   high_freq <- ll[which.max(ll$Freq),]
   
   if(as.numeric(high_freq[,1]) > 1) {
     as.numeric(levels(high_freq[,2]))[high_freq[,2]]
   } else {
     row[3]
   }
}

election <- apply(all_predicts,
                  1,
                  voting_booth)

caret::confusionMatrix(as.factor(election), reference = validate_set$accuracy_group)
```

Using an ensemble method, we see that we get back an overall better accuracy at __54.09%__. Compared to SVM using gain ratio, we gained roughly 9% on group 0, bringing us in range of our other predictions. We did however lose a bit of accuracy (~1%) on group 3 as compared to our SVM which had the highest accuracy on group 3 overall at 94.98%. Accuracy groups 1 and 2 are still very poorly identified in part due to using SVM as our default value if there is no consensus. Groups 1 and 2 were poorly defined across all models, however.

Different combinations were used to determine the best voting outcome. The best outcome was outlined above.

Also, using KNN or RF as the fallback selection gave a better spread of accuracy group predictions, but at a slightly lower accuracy score.

# __Conclusions__

While 54.09% appears quite low, the top Kaggle public score currently is ~57%. Which makes our models not too awful.

It certainly appears that game time and accuracy over the course of the games played for the PBS Measure UP! application contribute to the final assessment outcomes. However, it would also appear that there are other factors that contribute to an individuals ability to do well on a given assessment. PBS offers another file, the specs.csv file, that was not utilized in this analysis. This file could likely contain pertinent information about a child's ability to learn and retain information. Specifically the file shows that the __event_code__ attribute may offer a means of extracting such information. 

Outside of the information provided by PBS, there are likely other environmental factors that we are not taking into account, nor would we be likely to through the data gained from an application such as this.

Continued exploration into the specs.csv file alongside more robust feature engineering is certainly required to gain a more complete understanding of the factors that we are able to learn with the information provided.

```{r}
stopCluster(cl)
```







