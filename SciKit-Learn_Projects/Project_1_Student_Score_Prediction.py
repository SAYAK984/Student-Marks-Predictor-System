import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

df = pd.read_csv("I:\ML_FOLDER\SciKit-Learn_FOLDER\SciKit-Learn_Projects\Students Performance Dataset.csv")

#1
# Dataset preview........
print(f"First 10 rows of the DataFrame......\n{df.head(10)}\n")


#2
# Dataset Pre-Processing (Part-1)......

#2.1
# Information about the DataFrame........
print(f"Information about the given dataframe.......\n{df.info()}\n")
"""
Attributes Present in the DataFrame........

Midterm_Score,
Assignments_Avg,
Quizzes_Avg,
Participation_Score,
Projects_Score,

Attendance (%),
Study_Hours_per_Week,
Sleep_Hours_per_Night
Stress_Level (1-10),
Age,

Gender,
Department,
Extracurricular_Activities,
Internet_Access_at_Home,
Parent_Education_Level,
Family_Income_Level,

Final_Score,

Total_Score,
"""


#2.2
# Checking presence of NaN/Null-values.......
print(f"Checking the presence of NaN/Null-values in each column (if any).....\n{df.isna().sum()}\n")
"""
Attribute containing NaN and it's count.....

Parent_Education_Level        1025
"""
#Replacing the NaN/Null-values in 'Parent_Education_Level' with High School.........
df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("High School")

print(f"Now, presence of NaN/Null-values in each column (if any).....\n{df.isna().sum()}\n")
"""
Now, No attribute containing NaN values left........
"""


#2.3
# Checking presence of infinity.....
print(f"Checking the presence of infinity in each column (if any).....\n{df.isin([-np.inf, np.inf]).sum()}\n")
"""
#No infinity present in any attribute......
"""


#2.4
# Making a new DataFrame containing only relevant Features and Targets..........
df_new = df[["Student_ID",
             "Midterm_Score",
             "Assignments_Avg",
             "Quizzes_Avg",
             "Participation_Score",
             "Projects_Score",
             "Attendance (%)",
             "Study_Hours_per_Week",
             "Sleep_Hours_per_Night",
             "Stress_Level (1-10)",
             "Age",
             "Final_Score", 
             "Total_Score"]].copy()

print(f"First 5 rows of the new DataFrame (subset of the original one)......\n{df_new.head()}\n")


#3
# Dataset Pre-Processing (Part-2)......

#3.1
# Label-Encoding of Binary-Categorical Features ("Gender", "Extracurricular_Activities", Internet_Access_at_Home").........
LEobj = LabelEncoder()
df_new["Gender_Encoded"] = LEobj.fit_transform(df["Gender"])
df_new["Extracurricular_Activities_Encoded"] = LEobj.fit_transform(df["Extracurricular_Activities"])
df_new["Internet_Access_at_Home_Encoded"] = LEobj.fit_transform(df["Internet_Access_at_Home"])


#3.2
# One Hot-Encoding of Non-Binary Categorical Features ("Department", "Parent_Education_Level", "Family_Income_Level").........
OHEobj = OneHotEncoder(sparse_output = False)
encoded_array = OHEobj.fit_transform(df[["Department", "Parent_Education_Level", "Family_Income_Level"]]).astype(int)
encoded_df = pd.DataFrame(encoded_array, columns = OHEobj.get_feature_names_out(["Department", "Parent_Education_Level", "Family_Income_Level"]))


print(OHEobj.get_feature_names_out(["Department", "Parent_Education_Level", "Family_Income_Level"]),"\n")
"""
['Department_Business' 'Department_CS' 'Department_Engineering'
 'Department_Mathematics' 'Parent_Education_Level_Bachelor's'
 'Parent_Education_Level_High School' 'Parent_Education_Level_Master's'
 'Parent_Education_Level_PhD' 'Family_Income_Level_High'
 'Family_Income_Level_Low' 'Family_Income_Level_Medium']
"""


print(encoded_df.head(10),"\n")
encoded_df.insert(0, "Student_ID", df["Student_ID"])

df_new = pd.merge(df_new, encoded_df, on = "Student_ID", how = "outer")

print(f"First 5 rows of the new DataFrame (after merging/ encoded-attributes included)......\n{df_new.head()}\n")

#df_new.to_csv("I:/ML_FOLDER/SciKit-Learn_FOLDER/SciKit-Learn_Projects/Student_DataFrame(encoded-version).csv", index = False)



#4
# Features-Target Separation..........

#4.1
# Features-Target Separation for Model-1 (Predicting Final_Score)................
X1 = df_new[["Student_ID", "Midterm_Score",
         "Assignments_Avg",
         "Quizzes_Avg",
         "Participation_Score",
         "Projects_Score",
         "Attendance (%)",
         "Study_Hours_per_Week",
         "Sleep_Hours_per_Night",
         "Stress_Level (1-10)",
         "Age",
         "Gender_Encoded",
         "Extracurricular_Activities_Encoded",
         "Internet_Access_at_Home_Encoded",
         "Department_Business", 
         "Department_CS", 
         "Department_Engineering",      
         "Department_Mathematics",
         "Parent_Education_Level_Bachelor's",
         "Parent_Education_Level_High School",
         "Parent_Education_Level_Master's",
         "Parent_Education_Level_PhD", 
         "Family_Income_Level_High",
         "Family_Income_Level_Low", 
         "Family_Income_Level_Medium"]]   # 24 Features.....

y1 = df_new["Final_Score"]


#4.2
# Features-Target Separation for Model-2 (Predicting Total_Score)............
X2 = df_new[["Midterm_Score",
         "Final_Score",
         "Assignments_Avg",
         "Quizzes_Avg",
         "Participation_Score",
         "Projects_Score",
         "Extracurricular_Activities_Encoded"]]   # 7 Features.....

y2 = df_new["Total_Score"]



#5
# Train-Test Split for both Model-1 and Model-2.......

#5.1
# For Model-1..........
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 42)

#5.2
# For Model-2..........
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 42)

print(X1_train.head())

print(X2_test.head())
print("\n")


#6
# Standard-Scaling of the numeric-attributes......

#6.1
# Scaling of 1st Dataset.......
numeric_columns_1 = ["Midterm_Score", "Assignments_Avg", "Quizzes_Avg", "Participation_Score", "Projects_Score",
         "Attendance (%)", "Study_Hours_per_Week", "Sleep_Hours_per_Night", "Stress_Level (1-10)", "Age"]

scaler_1 = StandardScaler()
scaler_1.fit(X1_train[numeric_columns_1])
X1_train[numeric_columns_1] = scaler_1.transform(X1_train[numeric_columns_1])
X1_test[numeric_columns_1] = scaler_1.transform(X1_test[numeric_columns_1])

#6.2
# Scaling of 2nd Dataset.......
numeric_columns_2 = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg", "Participation_Score", "Projects_Score"]
scaler_2 = StandardScaler()
scaler_2.fit(X2_train[numeric_columns_2])
X2_train[numeric_columns_2] = scaler_2.transform(X2_train[numeric_columns_2])
X2_test[numeric_columns_2] = scaler_2.transform(X2_test[numeric_columns_2])

print(f"Scaled-form of the first-Training dataset......\n{X1_train}\n")

print(f"Scaled-form of the second-Testing dataset......\n{X2_test}\n")




#7
# Reducing the dimensionality of the Feature-sets with the help of PCA.......

#7.1 
# For 1st Feature-Set (Training and Testing)..........
categorical_columns_1 = ["Student_ID", "Gender_Encoded", "Extracurricular_Activities_Encoded",
         "Internet_Access_at_Home_Encoded",
         "Department_Business", "Department_CS", "Department_Engineering", "Department_Mathematics",
         "Parent_Education_Level_Bachelor's", "Parent_Education_Level_High School", "Parent_Education_Level_Master's",
         "Parent_Education_Level_PhD", 
         "Family_Income_Level_High", "Family_Income_Level_Low", "Family_Income_Level_Medium"]

obj1 = PCA()
obj1.fit(X1_train[numeric_columns_1])

print(obj1.explained_variance_ratio_)
print(obj1.explained_variance_ratio_.cumsum())
print("\n")

# 1st Training Feature-Set........
X1_train_reduced = obj1.transform(X1_train[numeric_columns_1])
X1_train_reduced_df = pd.DataFrame(X1_train_reduced, columns = [f"PC{x}" for x in range(obj1.n_components_)], index = X1_train.index)
X1_train_reduced_df.insert(0, "Student_ID", X1_train["Student_ID"])
X1_train_reduced_df = pd.merge(X1_train_reduced_df, X1_train[categorical_columns_1], on = "Student_ID", how = "outer")
X1_train_reduced_df.drop(columns = ["Student_ID"], inplace = True)
print(f"PCA reduced 1st input-Feature (Training-Data).......\n{X1_train_reduced_df.head()}\n")

# 1st Testing Feature-Set.......
X1_test_reduced = obj1.transform(X1_test[numeric_columns_1])
X1_test_reduced_df = pd.DataFrame(X1_test_reduced, columns = [f"PC{x}" for x in range(obj1.n_components_)], index = X1_test.index)
X1_test_reduced_df.insert(0, "Student_ID", X1_test["Student_ID"])
X1_test_reduced_df = pd.merge(X1_test_reduced_df, X1_test[categorical_columns_1], on = "Student_ID", how = "outer")
X1_test_reduced_df.drop(columns = ["Student_ID"], inplace = True)
print(f"PCA reduced 1st input-Feature (Testing-Data).......\n{X1_test_reduced_df.head()}\n")


print("Checking the Presence of NaNs in all types of Feature-sets......\n")
print(f"For 1st Dataset (Training)......\n {X1_train_reduced_df.isna().sum()}\n")
print(f"For 1st Dataset (Testing).......\n {X1_test_reduced_df.isna().sum()}\n")


# Finding out the correlationships with the Input-Features and the Target-Attributes.......
corr = df_new.corr(numeric_only=True)
print(f"{corr['Final_Score'].sort_values(ascending=False)}\n")

corr = df_new.corr(numeric_only=True)
print(f"{corr['Total_Score'].sort_values(ascending=False)}\n")



#8
# Model Training and Testing............

#8.1
# For Model-1 (Predicting Final_Score).........
model_1 = LinearRegression()
model_1.fit(X1_train_reduced_df, y1_train)
y1_predicted = model_1.predict(X1_test_reduced_df)

#8.2
# For Model-2 (Predicting Total_Score).........
model_2 = LinearRegression()
model_2.fit(X2_train, y2_train)
y2_predicted = model_2.predict(X2_test)




#9
# Model-Evaluation......

#9.1
# For Model-1..............
print(f"Evaluation of Model-1 (Final score prediction for a Student)........")
print(f"Mean Absolute Error = {round(mean_absolute_error(y1_test, y1_predicted), 3)}\n")
print(f"Mean Squared Error = {round(mean_squared_error(y1_test, y1_predicted), 3)}\n")
print(f"Root Mean Squared Error = {round(root_mean_squared_error(y1_test, y1_predicted), 3)}\n")
print(f"r^2 score = {round(r2_score(y1_test, y1_predicted), 3)}\n")

#9.2
# For Model-2..............
print(f"Evaluation of Model-2 (Total score prediction for a Student)........")
print(f"Mean Absolute Error = {round(mean_absolute_error(y2_test, y2_predicted), 3)}\n")
print(f"Mean Squared Error = {round(mean_squared_error(y2_test, y2_predicted), 3)}\n")
print(f"Root Mean Squared Error = {round(root_mean_squared_error(y2_test, y2_predicted), 3)}\n")
print(f"r^2 score = {round(r2_score(y2_test, y2_predicted), 3)}\n")



#10
# Visualization of the Target-data.....
"""
plt.hist(np.array(y1).ravel(), bins = 30, color = "yellow", edgecolor = "black", label = "Total students at a range")

plt.title("Distribution of students w.r.t their marks")
plt.xlabel("Distribution of Marks ------------>")
plt.ylabel("No. of students ------------>")

plt.legend(loc = "upper right", fontsize = 9)
plt.grid(color = "gray", linestyle = ":", linewidth = 0.5)

"""

fig, axes = plt.subplots(1, 2, figsize = (10, 8))


# X-axis values
x = X1_test["Study_Hours_per_Week"]

# Sort indices
sorted_idx = np.argsort(x)

# Sorted values
x_sorted = x.iloc[sorted_idx]
y_actual_sorted = y1_test.iloc[sorted_idx]
y_pred_sorted = y1_predicted[sorted_idx]

axes[0].scatter(x_sorted, y_actual_sorted, color="yellow", label="Actual Score of a Student")
axes[0].plot(x_sorted[::10], y_pred_sorted[::10], color="red", linestyle = "-", linewidth=2, label="Predicted score of a student (Regression Line)")

axes[0].set_title("Predicted Score Vs Actual Score (For Model-1)")
axes[0].set_xlabel("Study Hours per Week -------------->")
axes[0].set_ylabel("Total Score of the Student (Actual and Predicted) ------------>")

axes[0].legend(loc = "upper right", fontsize = 9)
axes[0].grid(color = "gray", linestyle = ":", linewidth = 0.5)



x2 = X2_test["Final_Score"]

sorted_idx_2 = np.argsort(x2)

x_sorted_2 = x2.iloc[sorted_idx_2]
y_actual_sorted_2 = y2_test.iloc[sorted_idx_2]
y_pred_sorted_2 = y2_predicted[sorted_idx_2]

axes[1].scatter(x_sorted_2, y_actual_sorted_2, color = "yellow", label = "Actual Score of a Student")
axes[1].plot(x_sorted_2[::10], y_pred_sorted_2[::10], color = "red", linestyle = "-", linewidth = 2, label = "Predicted score of a student (Regression Line)")

axes[1].set_title("Predicted Score Vs Actual Score (For Model-2)")
axes[1].set_xlabel("Final Score of the Student ------------>")
axes[1].set_ylabel("Total Score of the Student (Actual and Predicted) ------------>")

axes[1].legend(loc = "upper right", fontsize = 9)
axes[1].grid(color = "gray", linestyle = ":", linewidth = 0.5)


plt.tight_layout()
plt.show()

