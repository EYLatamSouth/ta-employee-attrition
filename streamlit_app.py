import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from imblearn.over_sampling import SMOTE


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import SessionState


def get_classification_report(y_test, y_pred):
    '''Source: https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format'''
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(
        by=['f1-score'], ascending=False)
    return df_classification_report


'''
# Retenção de Funcionários

Este conjunto de dados foi disponibilizado pela IBM e pode ser acessado neste [link](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset) do Kaggle.

O desenvolvimento do modelo e análise de features tem como referência este [notebook](https://www.kaggle.com/arthurtok/employee-attrition-via-ensemble-tree-based-methods) disponível também no Kaggle.

Os dados são fictícios.

Os campos de avaliação seguem estas definições:
'''

col1, col2 = st.beta_columns(2)

with col1:
    '''
    Education

    * 1 Below College
    * 2 College
    * 3 Bachelor
    * 4 Master
    * 5 Doctor

    EnvironmentSatisfaction

    * 1 Low
    * 2 Medium
    * 3 High
    * 4 Very High

    JobInvolvement

    * 1 Low
    * 2 Medium
    * 3 High
    * 4 Very High

    JobSatisfaction

    * 1 Low
    * 2 Medium
    * 3 High
    * 4 Very High
    '''

with col2:
    '''
    PerformanceRating

    * 1 Low
    * 2 Good
    * 3 Excellent
    * 4 Outstanding

    RelationshipSatisfaction

    * 1 Low
    * 2 Medium
    * 3 High
    * 4 Very High

    WorkLifeBalance

    * 1 Bad
    * 2 Good
    * 3 Better
    * 4 Best
    '''

'''
## Conjunto de Dados

'''

attrition = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

attrition = attrition.drop(['EmployeeCount'], axis=1)
attrition = attrition.drop(['EmployeeNumber'], axis=1)
attrition = attrition.drop(['StandardHours'], axis=1)
attrition = attrition.drop(['Over18'], axis=1)

attrition


'''
## Análise Exploratória Rápida


'''

st.vega_lite_chart(attrition, {
    "mark": "rect",
    "width": 700,
    "height": 300,
    "encoding": {
        "x": {
            "bin": {"maxbins": 60},
            "field": "Age",
            "type": "quantitative"
        },
        "y": {
            "bin": {"maxbins": 40},
            "field": "TotalWorkingYears",
            "type": "quantitative"
        },
        "color": {
            "aggregate": "count",
            "type": "quantitative"
        }
    },
    "config": {
        "view": {
            "stroke": "transparent"
        }
    }
}
)

st.vega_lite_chart(attrition, {
    "mark": "rect",
    "width": 700,
    "height": 300,
    "encoding": {
        "x": {
            "bin": {"maxbins": 60},
            "field": "Age",
            "type": "quantitative"
        },
        "y": {
            "bin": {"maxbins": 40},
            "field": "YearsInCurrentRole",
            "type": "quantitative"
        },
        "color": {
            "aggregate": "count",
            "type": "quantitative"
        }
    },
    "config": {
        "view": {
            "stroke": "transparent"
        }
    }
}
)

st.vega_lite_chart(attrition, {
    "mark": "rect",
    "width": 700,
    "height": 300,
    "encoding": {
        "x": {
            "bin": {"maxbins": 60},
            "field": "YearsAtCompany",
            "type": "quantitative"
        },
        "y": {
            "bin": {"maxbins": 40},
            "field": "JobSatisfaction",
            "type": "quantitative"
        },
        "color": {
            "aggregate": "count",
            "type": "quantitative"
        }
    },
    "config": {
        "view": {
            "stroke": "transparent"
        }
    }
}
)

st.vega_lite_chart(attrition, {
    "mark": "rect",
    "width": 700,
    "height": 300,
    "encoding": {
        "x": {
            "bin": {"maxbins": 60},
            "field": "WorkLifeBalance",
            "type": "quantitative"
        },
        "y": {
            "bin": {"maxbins": 40},
            "field": "JobSatisfaction",
            "type": "quantitative"
        },
        "color": {
            "aggregate": "count",
            "type": "quantitative"
        }
    },
    "config": {
        "view": {
            "stroke": "transparent"
        }
    }
}
)

# Define a dictionary for the target mapping
target_map = {'Yes': 1, 'No': 0}
# Use the pandas apply method to numerically encode our attrition target variable
attrition["Attrition_numerical"] = attrition["Attrition"].apply(
    lambda x: target_map[x])

# creating a list of only numerical values
numerical = [u'Age', u'DailyRate', u'DistanceFromHome',
             u'Education', u'EnvironmentSatisfaction',
             u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
             u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
             u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
             u'StockOptionLevel', u'TotalWorkingYears',
             u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
             u'YearsInCurrentRole', u'YearsSinceLastPromotion', u'YearsWithCurrManager']

plt.figure(figsize=(19, 15))

corrMatrix = attrition[numerical].corr()

'''
## Matriz de Correlação


Suporta a decisão de escolhas de atributos para serem utilizados no treinamento. Quanto mais correlação houver, melhor será para o modelo, o inverso também ocorre, pois ao abrir mão de atributos que não contribuem para o aprendizado para o modelo, ele ficará mais preciso e mais leve.

'''

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.heatmap(corrMatrix)
# Use Matplotlib to render seaborn
st.pyplot()

'## Atributos Utilizados no Treinamento '

# Drop the Attrition_numerical column from attrition dataset first - Don't want to include that
attrition = attrition.drop(['Attrition_numerical'], axis=1)

# Empty list to store columns with categorical data
categorical = []
for col, value in attrition.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attrition.columns.difference(categorical)
categorical.remove('Attrition')

atributes_numerical = st.multiselect(
    'Atributos Numéricos Selecionados', numerical.values.tolist(), numerical.values.tolist())
atributes_categorical = st.multiselect(
    'Atributos Categóricos Selecionados', categorical, categorical)

# Store the categorical data in a dataframe called attrition_cat
attrition_cat = attrition[categorical]
attrition_cat = pd.get_dummies(attrition_cat)

# Store the numerical features to a dataframe attrition_num
attrition_num = attrition[atributes_numerical]

# Concat the two dataframes together columnwise
attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)

# Define a dictionary for the target mapping
target_map = {'Yes': 1, 'No': 0}
# Use the pandas apply method to numerically encode our attrition target variable
target = attrition["Attrition"].apply(lambda x: target_map[x])

session_state = SessionState.get(trained=False, train=None)

seed = 1   # We set our random seed to zero for reproducibility
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': seed,
    'verbose': 0
}

rf = RandomForestClassifier(**rf_params)

# Split data into train and test sets as well as for validation and testing
train, test, target_train, target_val = train_test_split(attrition_final,
                                                         target,
                                                         train_size=0.80,
                                                         random_state=1)
oversampler = SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_sample(train, target_train)

if st.button('Treinar Modelo') or session_state.trained:
    'Iniciando o treinamento...'

    rf.fit(smote_train, smote_target)

    session_state.model = rf
    session_state.trained = True

    'Treinamento terminado.'
    'Verificando predições...'

    rf_predictions = rf.predict(test)

    st.success('Modelo treinado e valido com sucesso!')
    score = "Pontuação de Precisão (Accurácia): {}".format(
        accuracy_score(target_val, rf_predictions))
    session_state.score = score

    st.info(score)

    plt.figure(figsize=(19, 15))
    (pd.Series(rf.feature_importances_, index=attrition_final.columns.values)
     .nlargest(20)
     .plot(kind='barh'))
    st.pyplot()

    test_value = [20, 2000, 10, 5, 2, 94, 3, 4, 4, 5993, 19479, 8, 11, 3, 1, 0, 8, 0, 1, 6, 4,
                  0, 5, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]

    col2_1, col2_2, col2_3 = st.beta_columns(3)

    with col2_1:
        s_business_travel = st.selectbox(
            "Business Travel", attrition["BusinessTravel"].unique())

        if s_business_travel == 'Non_Travel':
            test_value[23] = 1
            test_value[24] = 0
            test_value[25] = 0
        elif s_business_travel == 'Travel_Frequently':
            test_value[23] = 0
            test_value[24] = 1
            test_value[25] = 0
        else:
            test_value[23] = 0
            test_value[24] = 0
            test_value[25] = 1

        s_department = st.selectbox(
            "Department", attrition["Department"].unique())

        if s_department == 'Human Resources':
            test_value[26] = 1
            test_value[27] = 0
            test_value[28] = 0
        elif s_business_travel == 'Research & Development':
            test_value[26] = 0
            test_value[27] = 1
            test_value[28] = 0
        else:
            test_value[26] = 0
            test_value[27] = 0
            test_value[28] = 1

        s_education_field = st.selectbox(
            "Education Field", attrition["EducationField"].unique())

        if s_education_field == 'Human Resources':
            test_value[29] = 1
            test_value[30] = 0
            test_value[31] = 0
            test_value[32] = 0
            test_value[33] = 0
            test_value[34] = 0
        elif s_business_travel == 'Life Sciences':
            test_value[29] = 0
            test_value[30] = 1
            test_value[31] = 0
            test_value[32] = 0
            test_value[33] = 0
            test_value[34] = 0
        elif s_business_travel == 'Marketing':
            test_value[29] = 0
            test_value[30] = 0
            test_value[31] = 1
            test_value[32] = 0
            test_value[33] = 0
            test_value[34] = 0
        elif s_business_travel == 'Medical':
            test_value[29] = 0
            test_value[30] = 0
            test_value[31] = 0
            test_value[32] = 1
            test_value[33] = 0
            test_value[34] = 0
        elif s_business_travel == 'Other':
            test_value[29] = 1
            test_value[30] = 0
            test_value[31] = 0
            test_value[32] = 0
            test_value[33] = 1
            test_value[34] = 0
        else:
            test_value[29] = 0
            test_value[30] = 0
            test_value[31] = 0
            test_value[32] = 0
            test_value[33] = 0
            test_value[34] = 1

    with col2_2:
        s_gender = st.selectbox("Gender", attrition["Gender"].unique())

        if s_gender == 'Human Resources':
            test_value[35] = 1
            test_value[36] = 0
        else:
            test_value[35] = 0
            test_value[36] = 1

        s_job_role = st.selectbox("Job Role", attrition["JobRole"].unique())

        if s_job_role == 'Healthcare Representative':
            test_value[37] = 1
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Human Resources':
            test_value[37] = 0
            test_value[38] = 1
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Laboratory Technician':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 1
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Manager':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 1
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Manufacturing Director':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 1
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Research Director':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 1
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Research Scientist':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 1
            test_value[44] = 0
            test_value[45] = 0
        elif s_job_role == 'Sales Executive':
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 1
            test_value[45] = 0
        else:
            test_value[37] = 0
            test_value[38] = 0
            test_value[39] = 0
            test_value[40] = 0
            test_value[41] = 0
            test_value[42] = 0
            test_value[43] = 0
            test_value[44] = 0
            test_value[45] = 1

        s_marital_status = st.selectbox(
            "Marital Status", attrition["MaritalStatus"].unique())

        if s_marital_status == 'Divorced':
            test_value[46] = 1
            test_value[47] = 0
            test_value[48] = 0
        elif s_business_travel == 'Married ':
            test_value[46] = 0
            test_value[47] = 1
            test_value[48] = 0
        else:
            test_value[46] = 0
            test_value[47] = 0
            test_value[48] = 1

    with col2_3:

        s_over_time = st.selectbox("Over Time", attrition["OverTime"].unique())

        if s_over_time == 'No':
            test_value[49] = 1
            test_value[50] = 0
        else:
            test_value[49] = 0
            test_value[50] = 1

    col3_1, col3_2 = st.beta_columns(2)

    with col3_1:
        n_age = st.slider("Age", int(attrition["Age"].min()), int(
            2*attrition["Age"].max()), test_value[0])
        test_value[0] = n_age

        n_daily_rate = st.slider("Daily Rate", int(attrition["DailyRate"].min()), int(
            2*attrition["DailyRate"].max()), test_value[1])
        test_value[1] = n_daily_rate

        n_dist_home = st.slider("Distance from Home", int(attrition["DistanceFromHome"].min(
        )), int(2*attrition["DistanceFromHome"].max()), test_value[2])
        test_value[2] = n_dist_home

        n_education = st.slider("Education", int(attrition["Education"].min()), int(
            2*attrition["Education"].max()), test_value[3])
        test_value[3] = n_education

        n_env_satisf = st.slider("Environment Satisfaction", int(attrition["EnvironmentSatisfaction"].min(
        )), int(attrition["EnvironmentSatisfaction"].max()), test_value[4])
        test_value[4] = n_env_satisf

        n_hour_rate = st.slider("Hourly Rate", int(attrition["HourlyRate"].min()), int(
            2*attrition["HourlyRate"].max()), test_value[5])
        test_value[5] = n_hour_rate

        n_job_involv = st.slider("Job Involvement", int(attrition["JobInvolvement"].min(
        )), int(attrition["JobInvolvement"].max()), test_value[6])
        test_value[6] = n_job_involv

        n_job_level = st.slider("Job Level", int(attrition["JobLevel"].min()), int(attrition["JobLevel"].max()) test_value[7])
        test_value[7] = n_job_level

        n_job_satisf = st.slider("Job Satisfaction", int(attrition["JobSatisfaction"].min(
        )), int(attrition["JobSatisfaction"].max()), test_value[8])
        test_value[8] = n_job_satisf

        n_month_income = st.slider("Monthly Income", int(attrition["MonthlyIncome"].min(
        )), int(2*attrition["MonthlyIncome"].max()), test_value[9])
        test_value[9] = n_month_income

        n_monthy_rate = st.slider("Monthly Rate", int(attrition["MonthlyRate"].min()), int(
            2*attrition["MonthlyRate"].max()), test_value[10])
        test_value[10] = n_monthy_rate

        num_comp_work = st.slider("Num. Companies Worked", int(
            2*attrition["NumCompaniesWorked"].min()), int(attrition["NumCompaniesWorked"].max()), test_value[11])
        test_value[11] = num_comp_work

    with col3_2:

        sal_hike = st.slider("% Salary Hike", int(
            attrition["PercentSalaryHike"].min()), int(2*attrition["PercentSalaryHike"].max()), test_value[12])
        test_value[12] = sal_hike

        n_perf_rating = st.slider("Performance Rating", int(attrition["PerformanceRating"].min(
        )), int(attrition["PerformanceRating"].max()), test_value[13])
        test_value[13] = n_perf_rating

        n_relat_satisf = st.slider("Relationship Satisfaction", int(attrition["RelationshipSatisfaction"].min(
        )), int(attrition["RelationshipSatisfaction"].max()), test_value[15])
        test_value[14] = n_relat_satisf

        n_stock_op = st.slider("Stock Option Level", int(
            attrition["StockOptionLevel"].min()), int(attrition["StockOptionLevel"].max()), test_value[15])
        test_value[15] = n_stock_op

        n_total_work_year = st.slider("Total Working Years", int(
            2*attrition["TotalWorkingYears"].min()), int(attrition["TotalWorkingYears"].max()), test_value[16])
        test_value[16] = n_total_work_year

        n_train_last_y = st.slider("Training Times Last Year", int(
            2*attrition["TrainingTimesLastYear"].min()), int(attrition["TrainingTimesLastYear"].max()), test_value[17])
        test_value[17] = n_train_last_y

        n_worklife_bal = st.slider("Work Life Balance", int(
            attrition["WorkLifeBalance"].min()), int(attrition["WorkLifeBalance"].max()), test_value[18])
        test_value[18] = n_worklife_bal

        n_years_comp = st.slider("Years at Company", int(attrition["YearsAtCompany"].min(
        )), int(2*attrition["YearsAtCompany"].max()), test_value[19])
        test_value[19] = n_years_comp

        n_years_role = st.slider("Years in Current Role", int(attrition["YearsInCurrentRole"].min(
        )), int(2*attrition["YearsInCurrentRole"].max()), test_value[20])
        test_value[20] = n_years_role

        n_years_promo = st.slider("Years since Last Promotion", int(attrition["YearsSinceLastPromotion"].min(
        )), int(2*attrition["YearsSinceLastPromotion"].max()), test_value[21])
        test_value[21] = n_years_promo

        n_years_manager = st.slider("Years with Current Manager", int(
            attrition["YearsWithCurrManager"].min()), int(2*attrition["YearsWithCurrManager"].max()), test_value[22])
        test_value[22] = n_years_manager

    test_value = [test_value]

    try:
        if rf.predict(test_value)[0] == 1:
            st.warning("Attrition")
        else:
            st.success("No Attrition")
            st.balloons()

        st.progress(int(100*rf.predict_proba(test_value)
                        [0][rf.predict(test_value)[0]]))

    except Exception as e:
        st.warning(e)
