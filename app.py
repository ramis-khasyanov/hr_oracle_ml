from flask import Flask
from flask import request
import pandas as pd
import os
import psycopg2
import json
import datetime
import joblib
from sqlalchemy import create_engine

user = os.environ['DB_USER']
password = os.environ['DB_PASSWORD']
host_ip = os.environ['DB_HOST_IP']
port = os.environ['DB_PORT']
database = "hroracledb"

engine_string = "postgresql://{user}:{password}@{host_ip}:{port}/{database}".format(user=user, password=password, host_ip=host_ip, port=port, database=database)
engine = create_engine(engine_string, echo=False, encoding='utf8')

relevant_data = [
    'e_position_eng',
    'e_salary_base',
    'u_department',
    'u_unit',
    'u_ntt',
    'e_age',
    'e_gender',
    'u_tl_tenure',
    'u_tl_age',
    'u_tl_gender',
    'u_hires_week',
    'u_exits_week',
    'u_hires_month',
    'u_exits_month',
    'u_headcount',
    'u_age_mean',
    'u_tenure_mean',
    'u_gender_average',
    'e_entrance_type',
    'e_source',
    'e_days_to_hire',
    'e_recomended',
    'r_age',
    'r_tenure',
    'r_level',
    'u_tl_active',
    'e_commute',
]

categorical_data = [
    'e_position_eng',
    'u_department',
    'u_unit',
    'e_source',
]

missing_values_mean = [
    'u_tl_tenure',
    'u_tl_age',
    'e_days_to_hire',
    'r_age',
    'r_tenure',
    'r_level',
    'e_commute',
]

initial_dummies = [
        'e_position_eng_DC_employee',
        'e_position_eng_Leading_DC_employee',
        'e_position_eng_Senior_DC_employee',
        'u_department_B2B',
        'u_department_Shift_1',
        'u_department_Shift_2',
        'u_department_Shift_3',
        'u_department_Shift_4',
        'u_department_Trouble_Shooting',
        'u_unit_B2B_Area',
        'u_unit_Inventarization_Area',
        'u_unit_Loading_Area',
        'u_unit_New_Arrivals_Area',
        'u_unit_Pack_Item_Area',
        'u_unit_Packing_Area',
        'u_unit_Picking_Area',
        'u_unit_Putaway_Area',
        'u_unit_Return_Area',
        'u_unit_Sorting_Area',
        'u_unit_Trouble_Area',
        'u_unit_Unpacking_Area',
        'e_source_headhunter',
        'e_source_job_mo',
        'e_source_lamoda',
        'e_source_other',
        'e_source_rabota',
        'e_source_ref',
        'e_source_superjob',
        'e_source_zarplata']


app = Flask(__name__)



@app.route("/test_predict", methods=['POST'])
def test_predict():
    json_str = request.get_json()
    data = json.loads(json_str)
    X_test = pd.Series(data).values.reshape(1, -1)
    model = joblib.load('models/our_model.pkl')
    y_pred = model.predict_proba(X_test)
    return str(float(y_pred[0][1]))

@app.route("/predict", methods=['POST'])
def predict():
    
    json_features = request.get_json()
    recieved_candidate_data = json.loads(json_features)
    candidate_df = pd.DataFrame(recieved_candidate_data, index=[1])
    recruiters_df = pd.read_sql_table("recruiters", con=engine)
    units_df = pd.read_sql_table("units", con=engine)
    units_df['e_id'] = recieved_candidate_data['e_id']
    df = pd.merge(candidate_df, recruiters_df, how="inner", on="e_recruiter")
    df = df.merge(units_df, how="outer", on="e_id")
    
    df['e_date_entered'] = pd.to_datetime(df['e_date_entered'])
    df['e_start_day'] = df['e_date_entered'].apply(lambda x: x.day)
    df['e_start_month'] = df['e_date_entered'].apply(lambda x: x.month)
    df['e_start_year'] = df['e_date_entered'].apply(lambda x: x.year)
    df['e_start_weekday'] = df['e_date_entered'].apply(lambda x: x.weekday())
    
    dummy_headers = []
    for cat_feature in categorical_data:
        dummy_columns = pd.get_dummies(df[cat_feature], prefix = cat_feature)
        dummy_headers = dummy_headers + list(dummy_columns.columns.values)
        df = pd.concat([df, dummy_columns], axis=1)
    
    for feature in missing_values_mean:
        mean_value = df[feature].mean()
        df.loc[:,feature].fillna(value = mean_value, inplace = True)
        
    for dummy_variable in initial_dummies:
        if dummy_variable not in dummy_headers:
            df[dummy_variable] = 0
            
    y_name = "e_probation"
    x_columns = [x for x in (relevant_data + initial_dummies) if (x != y_name or x in dummy_headers) and x not in categorical_data]
    x = df[x_columns]
    
    loaded_model = joblib.load("models/mybestmodel.pkl")
    loaded_model.predict(x)
    
    y_hat = loaded_model.predict_proba(x)
    
    Success_probability = []
    for i in y_hat:
        Success_probability.append(i[1])
        
    df = pd.concat([df, pd.DataFrame({'p_success_probability':Success_probability})], axis=1)
    df["e_date_entered"] = df["e_date_entered"].apply(lambda x: datetime.datetime.strftime(x, '%d.%m.%Y %H:%M'))
    
    candidate_predictions_to_send = df.to_json(orient='split')
    
    return json.dumps(candidate_predictions_to_send)


@app.route("/test", methods=["GET"])
def test_server():
    return 'Server is working well'


if __name__ == '__main__':
    app.run('0.0.0.0', 8080, debug=True)