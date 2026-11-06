# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone for the AI and ML Certification Program (E&ICT Academy, IIT Roorkee). The data is synthetic (`generate_data.py`); Delhivery is only the setting. All money is in INR.

## About the Project

This project is made as part of the AI and ML Certification Program by E&ICT Academy, IIT Roorkee.

The main aim of this project is to find out why parcels get delayed during delivery.

In this project, I have used Python and Machine Learning to:
- Predict how much time a parcel will take to reach.
- Find out whether a parcel may get delayed.
- Find delivery hubs where delays happen more often.
- Give suggestions to improve delivery.

**Note:** The data used in this project is created by a Python file called `generate_data.py`. It is not actual Delhivery company data. Delhivery is only used as an example. All money values are in INR.

## 1. How to Run -

First, install all the required things and then run the Python files one by one.

```bash
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

Run these files in the same order because some files need the results of the previous files.

**What each file does:**

- `generate_data.py` – Creates the data.
- `part_a_clean_eda.py` – Cleans the data and studies it.
- `part_b_regression.py` – Predicts delivery time.
- `part_c_classification.py` – Finds out if a parcel may be delayed.
- `part_d_segments.py` – Divides delivery hubs into different groups.
- `package_model.py` – Saves the final model.
- `predict_delay.py` – Checks new parcels and predicts possible delays.

The final model is saved in `models/delay_model.pkl`.

If the scikit-learn version is different, run `package_model.py` again to make a new model file.

## 2. What I Found in the Data

After cleaning and checking the data, I found some interesting things.

### 2.1 Delivery Service Type

There are three types of delivery: Express, Standard and Economy.

- Express: 119 out of 243 parcels were delayed.
- Standard: 20.56% delay rate.
- Economy: 8.57% delay rate.

Express has the most delays because it has a shorter delivery time limit of 36 hours.

### 2.2 Delivery Hub Performance

The hubs are divided into three groups called Tier 1, Tier 2 and Tier 3.

- Tier 1: 4% delays.
- Tier 2: 21.5% delays.
- Tier 3: 44% delays.

From this, we can see that Tier 3 hubs have more delays than the other hubs.

### 2.3 Delivery Distance

I also found that distance affects delivery time.

- 0–200 km: 2.19% delays.
- 201–400 km: 13.68% delays.
- 401–600 km: 43.64% delays.
- Above 600 km: 85.23% delays.

This shows that parcels travelling longer distances have more chances of getting delayed.

### 2.4 Main Things Affecting Delivery Time

The three main things related to delivery time are:

1. Promised delivery hours.
2. Distance travelled.
3. Number of delivery stops.

Out of these, promised delivery hours has the strongest connection with actual delivery time.

### 2.5 Log Transformation

Some parcels had very high values, which affected the average value of all parcels.

To make the data more balanced, log transformation was used. It helped make the data easier for the model to use.

### 2.6 Cleaning the Data

Before using the data, I made some changes:

- Removed 24 duplicate rows.
- Corrected 5 wrong weight values (999).
- Changed 12 different spellings of service types into 3 correct types.
- Final data contains 1,200 rows.
- Total delayed parcels: 278.
- Overall delay rate: 23.17%.

## 3. Predicting Delivery Time (Regression)

Regression is used to predict how many hours a parcel may take to reach its destination.

I tried different machine learning models to see which one gave better results.

### 3.1 Basic Model

First, I tried a simple model that predicted the average delivery time for every parcel.

It got an R² score of -0.0212, which means it did not give good results.

### 3.2 Comparing Different Models

I tested four models:

- Linear Regression
- Gradient Boosting
- Random Forest
- Decision Tree

Linear Regression gave the best result on the validation data.

Its results were:

- RMSE: 4.8912 hours.
- R² Score: 0.9313.

The final test results were:

- RMSE: 4.5235 hours.
- MAE: 3.5671 hours.
- R² Score: 0.9493.

These results show that the model was able to predict delivery time quite well on the test data.

### 3.3 Tree Depth

Tree depth decides how much a decision tree can learn from the data.

- Very small depth does not allow the model to learn enough.
- Very large depth can make the model learn the training data too closely.
- Depth 6 gave a better result for the Decision Tree model.

### 3.4 Learning Rate

Learning rate decides how quickly a model learns.

I tested three values: 0.01, 0.1 and 1.0.

The value 0.1 gave the best result for Gradient Boosting.

### 3.5 Cross-Validation

I also used cross-validation to check the models on different parts of the data.

Gradient Boosting got an average R² score of 0.9352, while Linear Regression got 0.9261.

Gradient Boosting performed slightly better in this test. However, Linear Regression was selected as the final model because it gave the lowest error on the validation data.

## 4. Finding Delayed Parcels (Classification)

Classification is used to find out whether a parcel will be delayed or not.

### 4.1 Basic Prediction

First, I tried a simple model that marked every parcel as not delayed.

It got 76.67% accuracy, but it could not find even one delayed parcel.

So, this model was not useful.

### 4.2 Selecting Important Columns

Some columns in the data were not useful for predicting delays.

After checking the data, I selected the 7 most useful columns.

Using these 7 columns gave the best F1 score of 0.8333 on the validation data.

### 4.3 Model Settings

I tried changing some settings of the Random Forest model, but the results did not improve.

So, I used the normal Random Forest settings:

- Number of trees: 100.
- Maximum depth: No limit.
- Minimum samples per leaf: 1.
- Selected columns: 7.

### 4.4 Delay Alert

The model gives a chance of whether a parcel will be delayed.

I selected 0.5 as the final value.

This means that if the delay chance is 50% or more, the parcel will be marked for an alert.

The validation results were:

- Precision: 79.55%.
- Recall: 83.33%.

### 4.5 Final Test Results

The final model gave these results:

- Precision: 68.89%.
- Recall: 75.61%.
- F1 Score: 72.09%.

The model correctly found 31 out of 41 delayed parcels in the test data.

This means that it was able to find around 75–80% of delayed parcels.

## 5. Delivery Hub Groups

I divided the delivery hubs into three groups according to how often their parcels get delayed.

### 5.1 Escalate Group

This group has 9 hubs with an average delay rate of 46%.

These hubs need more attention because they have more delivery delays.

The hubs are:

1. Nagpur – 46%.
2. Kochi – 36%.
3. Patna – 46%.
4. Ranchi – 38%.
5. Guwahati – 62%.
6. Raipur – 48%.
7. Bhubaneswar – 40%.
8. Siliguri – 44%.
9. Agartala – 54%.

### 5.2 Standard and Watch Groups

- Standard Group: 7 hubs with a 2.57% delay rate.
- Watch Group: 8 hubs with a 15.5% delay rate.

Standard hubs are doing well, while Watch hubs need regular checking.

### 5.3 Difference Between Hub Tier and Performance

I found that some hubs were not performing according to their given tier.

For example:

- Kolkata is Tier 1 but has a 14% delay rate.
- Nagpur and Kochi are Tier 2 but have more delays.
- Dehradun is Tier 3 but has a delay rate of 20%.

This means that we should check the actual performance of hubs instead of only looking at their tier.

### 5.4 How the Alert System Works

If a parcel has a 50% or higher chance of getting delayed, the model marks it for an alert.

According to the test results:

- Around 7 out of 10 alerted parcels were actually delayed.
- Around 8 out of 10 delayed parcels were found by the model.

This can help the delivery team take action earlier.

### 5.5 Checking New Parcels

I also tested the model on 48 new parcels.

The results were:

- 8 parcels were marked as possible delays.
- 6 parcels were from Escalate hubs and need attention from hub managers.
- 2 parcels need a call to the customer about the possible delay.

### 5.6 Why I Selected 3 Groups

I used K-Means to divide the hubs into different groups.

I tested 2, 3, 4 and 5 groups.

I selected 3 groups because:
- The groups were clearly different.
- All groups had a good number of hubs.
- Some groups became too small when I selected 4 or 5 groups.
- Three groups are also easier to manage.

## 6. Using the Model in the Future

If this project is used in a real delivery system, some things should be checked regularly.

### 6.1 Checking Results

The model's results can be checked using its test scores.

### 6.2 Correct Data

The data should be cleaned properly. Missing values, duplicate rows and wrong values should be checked.

### 6.3 Getting the Same Results

The project uses fixed settings so that it can give similar results when run again.

### 6.4 Speed

The model can check 48 new parcels in one run and can easily work on a normal laptop.

### 6.5 Easy to Update

The project has different Python files for different tasks. This makes it easier to understand and change.

### 6.6 Regular Checking

The results are saved in CSV files, so they can be checked whenever needed.

## 7. Things to Check Every Week

If the model is used in the future, these things should be checked every week:

1. **Delay Rate:** The current delay rate is 23.17%. If it goes below 18% or above 28% for two weeks, the reason should be checked.

2. **Alerted Parcels:** Normally, around 24–25% of parcels are marked for alerts. If this number changes too much, the model should be checked.

3. **Model Results:** If the model starts missing too many delayed parcels and its recall goes below 65%, it should be trained again.

### Updating the Model

The model should be updated every month with new delivery data.

Sale days should be compared with previous sale days because delays can increase during sales.

Changes in distance, delivery stops, missing data and hub performance should also be checked.

## 8. Limitations

This project has some limitations:

1. The data is created using Python and is not actual Delhivery data.
2. Real delivery problems can be more complicated than the problems in this project.
3. The test data has only 41 delayed parcels, so the results may change with more data.
4. The model was tested on a small amount of data.
5. There are only 24 delivery hubs in the project.
6. The data covers only 90 days, so it does not show all changes that may happen during the year.

## 9. Conclusion

Through this project, I learned how Python and Machine Learning can be used to solve delivery problems.

I found that distance, delivery type and hub performance can affect delivery delays.

I also tried different models to predict delivery time and find parcels that may arrive late.

The project also helped me find which delivery hubs need more attention.

Overall, this project gave me a basic understanding of how Machine Learning can help in finding problems and improving parcel delivery.

**Note:** All results are based on computer-generated data and do not show the actual performance of Delhivery.