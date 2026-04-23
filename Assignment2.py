import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, root_mean_squared_error, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, 
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


df = pd.read_csv('C:/Users/marcu/OneDrive/Desktop/BCIT/COMP 4254/Assignment2/' \
                    'archive/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Function used to compile results and complete test evaluation
def find_results(model, X_data, y_data, output_type='binary', x_scaler=None):
    #TODO: integrate code to handle scalers
    # applying scaler (optional)

    # this function will take the model, x and y data then test them though Kfold validation
    kf = KFold(n_splits=6, shuffle=True, random_state=123)

    store_scores = []
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

        if x_scaler != None:
            sc_x = x_scaler
            X_test = sc_x.fit_transform(X_test)
            X_train = sc_x.transform(X_train)

        testing_model = model
        testing_model.fit(X_train, y_train.values.ravel())

        predictions = testing_model.predict(X_test)
        # based on the type of data that we are predciting alot of it will be a binary decision however they might be a case we 
        # are looking at the types of addiction, we have to be flexible about the scoring

        # accuracy score can only handle classifier outputs, not possible for regression.
        accuracy = accuracy_score(predictions, y_test)
        recall = recall_score(predictions, y_test, average=output_type)
        precision = precision_score(predictions, y_test, average=output_type)
        f1 = f1_score(predictions, y_test, average=output_type)
        rsme = root_mean_squared_error(predictions, y_test)
        # storing the calculated results
        store_scores.append((accuracy, recall, precision, f1, rsme))
    
    #calculating the averages given the test results
    average_accuracy = np.mean(store_scores[0][:])
    average_recall = np.mean(store_scores[1][:])
    average_precision = np.mean(store_scores[2][:])
    average_f1 = np.mean(store_scores[3][:])
    average_rmse = np.mean(store_scores[4][:])

    # calculating the standard deciation based on the results of KFold
    stdev_accuracy = np.std(store_scores[0][:])
    stdev_recall = np.std(store_scores[1][:])
    stdev_precision = np.std(store_scores[2][:])
    stdev_f1 = np.std(store_scores[3][:])
    stdev_rmse = np.std(store_scores[4][:])

    # print_confusion(y_test, predictions)

    # print(f"Accuracy:   {average_accuracy} +/- {stdev_accuracy}")
    # print(f"Recall:     {average_recall} +/- {stdev_recall}")
    # print(f"Precision:  {average_precision} +/- {stdev_precision}")
    # print(f"F1 Score:   {average_f1} +/- {stdev_f1}")
    # print(f"RMSE:       {average_rmse} +/- {stdev_rmse}")

    return [average_accuracy, stdev_accuracy, average_recall, stdev_recall, average_precision, stdev_precision,
            average_f1, stdev_f1, average_rmse, stdev_rmse]

def print_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def showVIF(X): # need to rewrite this in my own code # this is a placeholder
    # this doesn't work becuase VIF score doesn't handle boolean well, convert true/false column to float
    # acedemic work impact no/ yes
    # genders as well
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("\nVariance Inflation Factors:")
    vif_df = vif_df.sort_values(["VIF"], ascending=False)
    print(vif_df)

def display_complied_reports(report_list):
    #columns = ['Acc', 'Acc Stdev', 'Recall', 'Recall Stdev', 'Prec', 'Prec Stdev', 'F1', 'F1 Stdev', 'RMSE', 'RMSE Stdev']
    print("MODEL \t ACC\tSTDEV\tREC\tSTDEV\tPREC\tSTDEV\tF1\tSTDEV\tRMSE\tSTDEV")
    for report in report_list:
        print(f"{report[0]:.4}\t{report[1]:.4}\t{report[2]:.4}\t{report[3]:.4}\t{report[4]:.4}\t{report[5]:.4}\t{report[6]:.4}\t{report[7]:.4}\t{report[8]:.4}\t{report[9]:.4}\t{report[10]:.4}")
        
def feature_selection(X, y, n=10):
    selector = SelectKBest(chi2, k=n)
    select_X = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {list(selected_features)}")

    X = X[selected_features]
    return X

def createStackingClassifier():
    # Define base models
    # base_models = [
    #     ('dt', DecisionTreeClassifier()),
    #     ('rf', RandomForestClassifier()),
    #     ('lr', LogisticRegression(solver='newton-cg')),
    #     ('et', ExtraTreesClassifier()),
    #     ('knn', KNeighborsClassifier()),
    #     ('svc', SVC(probability=True)),
    #     ('rg', RidgeClassifier()),
    #     ('en', LogisticRegression(penalty='elasticnet', solver='saga')),
    #     ('ada', AdaBoostClassifier())
    # ]
    base_models = [
        ('knn', KNeighborsClassifier(n_neighbors=4)),
        ('rf', RandomForestClassifier(max_depth=10, n_estimators=15)),
        ('lr', LogisticRegression(solver='newton-cg')),
        ('et', ExtraTreesClassifier()),
        ('rf2', RandomForestClassifier(max_depth=20, n_estimators=20)),
        ('lr2', LogisticRegression(solver='newton-cg')),
        ('rg', RidgeClassifier()),
        ('en', LogisticRegression(penalty='elasticnet', solver='saga') ),
        ('ada', AdaBoostClassifier())
    ]

    # Define meta model
    #meta_model = LogisticRegression(solver='newton-cg')
    meta_model = RidgeClassifier()

    # Create Stacking Classifier
    stack_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    return stack_classifier
    



print(df['addiction_level'].unique())
print(df['stress_level'].unique())

addiction_level_map = {'None': 0, 'Mild': 1, 'Moderate': 2,'Severe': 3}
df['addiction_level'] = df['addiction_level'].map(addiction_level_map)

stress_level_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['stress_level'] = df['stress_level'].map(stress_level_map)

df['addiction_level'].mask(((df['addiction_level'].isnull()) & (df['addicted_label']== 0)) , 0, inplace=True)
# we notice that there are no elements that violate the conditions that there is null values outside of the addicted_label == 0
# this implies that the Nan values were just the intended None values and that the dataset was complete
print(df['addiction_level'].unique())

df_final = pd.get_dummies(df, columns=['academic_work_impact', 'gender'])
print(df_final.head())


# print(df.describe())
y = df_final[['addicted_label']]
y2 = df_final[['addiction_level']]

X = df_final.copy()
del X['addicted_label']
del X['addiction_level']

# we are removing the transaction_id and user_id since that information is useless for this project
del X['transaction_id']
del X['user_id']

# converting bool features to type float
X['academic_work_impact_No'] = X['academic_work_impact_No'].astype(float)
X['academic_work_impact_Yes'] = X['academic_work_impact_Yes'].astype(float)
X['gender_Female'] = X['gender_Female'].astype(float)
X['gender_Male'] = X['gender_Male'].astype(float)
X['gender_Other'] = X['gender_Other'].astype(float)

# based on VIF scores we have to remove a few columns
del X['gender_Female']
del X['academic_work_impact_No']
del X['weekend_screen_time']

#looking at the VIF scores to identify multicolinearity
X = add_constant(X)
showVIF(X)

# # looking at the heatmap with removing the obvious multicolinearity
combine = pd.concat([X, y], axis=1)
correlation_matrix = combine.corr()
plt.figure(figsize=(10,8))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

X = feature_selection(X, y)

# output_type for find_result: binary, micro, macro, weighted
# only binary and micro should be necessary for this project
min_max = MinMaxScaler()
robust = RobustScaler()
standard = StandardScaler()

report_list1 = []
report_list2 = []
dt          = DecisionTreeClassifier()
lr          = LogisticRegression(solver='newton-cg')
rf          = RandomForestClassifier()
et          = ExtraTreesClassifier()
knn         = KNeighborsClassifier()
svc         = SVC()
rg          = RidgeClassifier()
en          = LogisticRegression(penalty='elasticnet', solver='saga')
ada         = AdaBoostClassifier()
stack       = createStackingClassifier()

model_dict = {"Decision Tree": dt, 
              "Logistic Regression": lr, 
              "Random Forest": rf, 
              "Extra Trees": et, 
              "KNieghbors": knn, 
              "SVC": svc, 
              "Ridge": rg, 
              "Elastic Net": en,
              "AdaBoost": ada,
              "Stacking": stack}

for key, value in model_dict.items():
    model_name = key
    print(f"++++++++++++++++{model_name}====================")

    temp_array = find_results(value, X, y, output_type='binary', x_scaler=min_max)
    temp_array.insert(0, f"{model_name}")
    report_list1.append(temp_array)

    temp_array = find_results(value, X, y2, output_type='micro', x_scaler=min_max)
    temp_array.insert(0, f"{model_name}")
    report_list2.append(temp_array)

display_complied_reports(report_list1)
print("")
display_complied_reports(report_list2)

# stack_classifier = StackingClassifier()
