from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from pydantic import BaseModel
import boto3
import pandas as pd
from prophet import Prophet
import datetime
from botocore.exceptions import ClientError
from datetime import datetime
from dateutil import relativedelta
from datetime import timedelta
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING, UpdateOne, ASCENDING
from urllib.parse import unquote
import numpy as np
import math
import neurokit2 as nk
import time

app = FastAPI()
load_dotenv()

class UserEmailRequest(BaseModel):
    user_email: str

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pnu-kkami2.vercel.app"],  # 프론트엔드 도메인
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


# AWS
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
TABLE_NAME = 'MyMentos'

dynamodb = boto3.client('dynamodb', 
                        aws_access_key_id=AWS_ACCESS_KEY,
                        aws_secret_access_key=AWS_SECRET_KEY,
                        region_name=AWS_REGION)



MONGODB_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGODB_URI)
db = client.get_database("heart_rate_db")
prediction_collection = db.prediction_results
analysis_collection = db.analysis_results
step_collection = db.step_results
sleep_collection = db.sleep_results
calorie_collection = db.calorie_results

####### 일 / 시간 #######
daily_collection = db.daily
hourly_collection = db.hourly


# raw data 저장 collection 및 query 240812
# daily hrv 저장 collection
bpm = db.bpm
steps = db.steps
calories = db.calories
sleeps = db.sleeps
hrv_collection = db.hrv


################ 데이터 각각 나눈 버전 ###############



################ 데이터 각각 나눈 버전 ###############
bpm_div = db.bpm_div
step_div = db.step_div
calorie_div = db.calorie_div
sleep_div = db.sleep_div

################ 데이터 저장 테스트 ##################
bpm_t = db.bpm_t
step_t = db.step_test
calorie_t = db.calorie_test
sleep_t = db.sleep_test


########## dynamodb process time check ##########
@app.post("/check_db3_dynamodb")
async def check_db_query_div_dynamodb(request: UserEmailRequest):
    
    # 전체 check DB 걸린 시간 -> all_end_time - all_start_time
    all_start_time = datetime.now()
    
    user_email = request.user_email
    record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
    collection_names_div = ['bpm_t', 'step_t', 'calorie_t', 'sleep_t']

    # MongoDB 컬렉션에 데이터가 존재하는지 걸린 시간 -> exist_items_end_time - exist_items_start_time
    exist_items_start_time = datetime.now()
    exist_times = exist_collection_div(user_email, collection_names_div)
    exist_items_end_time = datetime.now()
    print(f'In Python ---> MongoDB 컬렉션 데이터 존재 유무 체크 걸린 시간 @@ : {exist_items_end_time - exist_items_start_time}')
    
    # DynamoDB 데이터 query 걸린 시간 
    query_start_time = datetime.now()
    json_data = [new_query_div(user_email, record_names[x], exist_times[x]) for x in range(len(exist_times))]
    query_end_time = datetime.now()
    print(f'In Python ---> dynamoDB 데이터 query 걸린 시간 @@ : {query_end_time - query_start_time}')
    
    
    # DataFrame 만드는데 걸린 시간 
    create_df_start_time = datetime.now()
    df_data = [create_df_div(json_data[x]) for x in range(len(json_data))]
    create_df_end_time = datetime.now()
    print(f'In Python ---> dynamoDB 데이터 query 걸린 시간 @@ : {create_df_end_time - create_df_start_time}')
    
    mongo_save_start_time = datetime.now()
    [update_db_div(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))]
    mongo_save_end_time = datetime.now()
    print(f'In Python ---> MongoDB 저장 : {mongo_save_end_time - mongo_save_start_time} (2)')
    
    all_end_time = datetime.now()
    print(f'In Python ---> 전체 끝나는데 까지 걸린 시간 @@ : {all_end_time - all_start_time}')

########################################################





@app.post("/check_db3_div")
async def check_db_query_div(request: UserEmailRequest):
    check_db3_div_start_time = datetime.now()
    user_email = request.user_email
    record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
    collection_names_div = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
    
    dynamo_start_time = datetime.now()
    exist_times = exist_collection_div(user_email, collection_names_div)
    # print(f'exist_times ---> {exist_times}')
    json_data = [new_query_div(user_email, record_names[x], exist_times[x]) for x in range(len(exist_times))]
    # print(f'json_data ---> {json_data}')
    df_data = [create_df_div(json_data[x]) for x in range(len(json_data))]
    dynamo_end_time = datetime.now()
    print(f'DynamoDB Data 불러오기 및 DataFrame 생성 (1) : {dynamo_end_time - dynamo_start_time}s')
    
    mongo_start_time = datetime.now()
    [update_db_div(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))]
    mongo_end_time = datetime.now()
    print(f'MongoDB 저장 : {mongo_end_time - mongo_start_time} (2)')
    print(f'In Python ---> check_db3_div 끝나는데 까지 시간 @@ (1+2) : {mongo_end_time - check_db3_div_start_time}')
    
    
def exist_collection_div(user_email, collections):
    res_idx = []
    collection_list_name = db.list_collection_names()
    for idx in collections:
        if idx not in collection_list_name:
            db.create_collection(idx)
            if idx == 'sleep_div':
                db[idx].create_index([('user_email', ASCENDING), ('timestamp_start', ASCENDING)])
            else:
                db[idx].create_index([('user_email', ASCENDING), ('timestamp', ASCENDING)])
        else:
            if eval(idx).find_one({'user_email': user_email}) == None:
                res_idx.append('0000-00-00T00:00:00')
            else:
                if idx == 'sleep_div':
                    res_idx.append(str(eval(idx).find_one({'user_email':user_email}, sort=[('_id', DESCENDING)])['timestamp_end']))
                else:
                    res_idx.append(str(eval(idx).find_one({'user_email':user_email}, sort=[('_id', DESCENDING)])['timestamp']))
    return res_idx


def new_query_div(user_email, record_name, start_time):
    # 각 컬렉션 별 query 걸린 시간 체크...
    dynamodb_query_start_time = datetime.now()
    new_items = []
    start_time = start_time
    last_evaluated_key = None
    
    if record_name == 'Steps':
        start_time = start_time.replace(' ', 'T')
    else:
        if str(start_time)[:4] == '0000':
            start_time = start_time
        else:
            if record_name == 'HeartRate' or record_name == 'TotalCaloriesBurned':
                start_time = str(pd.to_datetime(start_time) - timedelta(hours=9) + timedelta(minutes=1)).replace(' ', 'T')
            else:
                start_time = str(pd.to_datetime(start_time) - timedelta(hours=9)).replace(' ', 'T')
    try:
        while True:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND SK BETWEEN :start_sk AND :end_sk',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':start_sk': {'S': f'{record_name}Record#{start_time}'},
                    ':end_sk': {'S': f'{record_name}Record#9999-12-31T23:59:59Z'},
                },
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
                
            response = dynamodb.query(**query_params)
            new_items.extend(response['Items'])
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
            
        dynamodb_query_end_time = datetime.now()
        print(f'{user_email} 사용자의 {record_name} Data 걸린 시간 체크 : {dynamodb_query_end_time - dynamodb_query_start_time}')
    
        return new_items

    except ClientError as e:
        print({e.response['Error']['Message']})
        return None

def create_df_div(query_json):
    if len(query_json) == 0:
        return []
    
    if 'HeartRate' in query_json[0]['SK']['S']:
        return pd.DataFrame({
            'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
            'bpm': [int(query_json[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(query_json))]
        })
        
    if 'Steps' in query_json[0]['SK']['S']:
        return pd.DataFrame({
            'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
            'step': [int(query_json[x]['recordInfo']['M']['count']['N']) for x in range(len(query_json))],
        })
        
    if 'TotalCaloriesBurned' in query_json[0]['SK']['S']:
        return pd.DataFrame({
            'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
            'calorie': [np.round(float(query_json[x]['recordInfo']['M']['energy']['M']['value']['N']), 3) for x in range(len(query_json))],
        })
    
    if 'SleepSession' in query_json[0]['SK']['S']:
        return pd.DataFrame({
            'ds_start': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
            'ds_end': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
            'sleep': [int(query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['stage']['N']) for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))],
        })
    
def prepare_docs(user_email, df, data_type):
    #print(f'hyobin: {data_type}')
    # print(f'hyobin: {df[:5]}')
    # print('in prepare_docs --> ', df)
    # print('data_type', data_type)
    # print("data_type[:data_type.find('_')]", data_type[:data_type.find('_')])
    # print('in prepare_docs ---- > : ', data_type)
    if data_type[:data_type.find('_')] == 'calorie':
        print('is in calorie Data ----- : ', df, len(df))
        return [
            {
                'user_email': user_email,
                'type': data_type[:data_type.find('_')],
                'value': float(row[data_type[:data_type.find('_')]]),
                'timestamp': row['ds'],
            }
            for row in df.to_dict('records')
        ]
        
    elif data_type[:data_type.find('_')] == 'sleep':
        return [
            {
                'user_email': user_email,
                'type': data_type[:data_type.find('_')],
                'value': int(row[data_type[:data_type.find('_')]]),
                'timestamp_start': row['ds_start'],
                'timestamp_end': row['ds_end'],
            }
            for row in df.to_dict('records')
        ]
        
    else: 
        return [
            {
                'user_email': user_email,
                'type': data_type[:data_type.find('_')],
                'value': int(row[data_type[:data_type.find('_')]]),
                'timestamp': row['ds']
            }
            for row in df.to_dict('records')
        ]

    

def update_db_div(user_email, df, collection):
    if len(df) == 0:
        return 0
    # prepare_docs() 구조 bpm, step, calorie, step 들어가야
    start_doc = time.time()
    documents = prepare_docs(user_email, df, collection)
    end_doc = time.time()
    # print(documents)
    print(f'{user_email} - {collection} prepare_docs 걸린 시간 : {end_doc - start_doc}')
    
    
    if collection[:collection.find('_')] == 'sleep':
        bulk_operations = [
            UpdateOne(
                {
                    'user_email': doc['user_email'],
                    'timestamp_start': doc['timestamp_start'],
                },
                {'$set':doc},
                upsert=True
            ) for doc in documents
        ]
        
    else:
        bulk_operations = [
            UpdateOne(
                {
                    'user_email': doc['user_email'],
                    'timestamp': doc['timestamp']
                },
                {'$set':doc},
                upsert=True
            ) for doc in documents
        ]
        
    batch_size = 5000
    total_operations = len(bulk_operations)
    
    start_time = time.time()
    
    for i in range(0, total_operations, batch_size):
        batch = bulk_operations[i:i+batch_size]
        eval(collection).bulk_write(batch, ordered=False)
        
    end_time = time.time()
    print(f'{user_email} 사용자의 {collection} Data 저장 걸린 시간 --> {end_time - start_doc}')

def calc_hrv(group_):
    rr_intervals = 60000/group_['bpm'].values
    if len(rr_intervals) == 1:
        return pd.Series({
            'rmssd': 0,
            'sdnn': 0,
        })
    peaks = nk.intervals_to_peaks(rr_intervals)
    hrv = nk.hrv_time(peaks)
    return pd.Series({
        'rmssd': np.round(hrv['HRV_RMSSD'].values[0], 3),
        'sdnn': np.round(hrv['HRV_SDNN'].values[0], 3),
    })
    
@app.get("/get_save_dates/{user_email}")
async def get_save_dates(user_email: str):
    collections = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
    # print('----234')
    # print('----234')
    # # print(list(str(max(exist_collection_div(user_email, collections)))))
    # print('----234')
    # print('----234')
    return {"save_dates": [max(exist_collection_div(user_email, collections))]}

@app.get("/get_save_dates_div/{user_email}")
async def get_save_dates_div(user_email: str):
    collections = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
    # print('----234')
    # print('----234')
    # # print(list(str(max(exist_collection_div(user_email, collections)))))
    # print('----234')
    # print('----234')
    return {"save_dates": [max(exist_collection_div(user_email, collections))]}

@app.get("/feature_hour_div/{user_email}/{start_date}/{end_date}")
async def bpm_hour_feature(user_email: str, start_date: str, end_date: str):
    query = bpm_div.find({'user_email': user_email, 'timestamp': {'$gte': datetime.fromtimestamp(int(str(start_date)[:-3])), '$lte': datetime.fromtimestamp(int(str(end_date)[:-3]))}})
    list_query = list(query)
    mongo_bpm_df = pd.DataFrame({
        'ds': [list_query[x]['timestamp'] for x in range(len(list_query))],
        'bpm': [list_query[x]['value'] for x in range(len(list_query))]
    })
    print(f'In featureHour len bpm_df {len(mongo_bpm_df)}')
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')
    mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

    hour_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=180)]
    hour_group = hour_df.groupby(hour_df['ds'].dt.floor('h'))
    hour_hrv = hour_group.apply(calc_hrv).reset_index()
    
    hour_hrv.rename(columns={'rmssd' : 'hour_rmssd', 'sdnn' : 'hour_sdnn'}, inplace=True)
    
    return {'hour_hrv': hour_hrv[['ds', 'hour_rmssd', 'hour_sdnn']].to_dict('records')}
     
@app.get("/feature_day_div/{user_email}")
async def bpm_day_feature(user_email: str):
    query = bpm_div.find({'user_email': user_email})
    list_query = list(query)
    mongo_bpm_df = pd.DataFrame({
        'ds': [list_query[x]['timestamp'] for x in range(len(list_query))],
        'bpm': [list_query[x]['value'] for x in range(len(list_query))]
    })
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')
    mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

    day_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=730)]
    day_group = day_df.groupby(day_df['ds'].dt.floor('d'))
    day_hrv = day_group.apply(calc_hrv).reset_index()
    
    day_hrv.rename(columns={'rmssd' : 'day_rmssd', 'sdnn' : 'day_sdnn'}, inplace=True)

    return {'day_hrv': day_hrv[['ds', 'day_rmssd', 'day_sdnn']].to_dict('records')}

@app.get("/predict_minute_div/{user_email}")
async def bpm_minute_predict(user_email: str):
    query = bpm_div.find({'user_email': user_email})
    list_query = list(query)
    mongo_bpm_df = pd.DataFrame({
        'ds': [list_query[x]['timestamp'] for x in range(len(list_query))],
        'bpm': [list_query[x]['value'] for x in range(len(list_query))]
    })
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')    
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    mongo_bpm_df = mongo_bpm_df.astype({'y': 'int32'})

    min_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=7)]

    min_model = Prophet(
        changepoint_prior_scale=0.0001,
        seasonality_mode='multiplicative',
    )
    min_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    min_model.add_country_holidays(country_name='KOR')
    min_model.fit(min_df)
    
    min_future = min_model.make_future_dataframe(periods=60*24*1, freq='min')
    min_forecast = min_model.predict(min_future)

    min_forecast.rename(columns={'yhat': 'min_pred_bpm'}, inplace=True)
    min_forecast['min_pred_bpm'] = np.round(min_forecast['min_pred_bpm'], 3)  
    min_forecast = min_forecast[len(min_forecast) - 60*24*1:]

    return {'min_pred_bpm': min_forecast[['ds', 'min_pred_bpm']].to_dict('records')}
       
@app.get("/predict_hour_div/{user_email}")
async def bpm_hour_predict(user_email: str):
    query = bpm_div.find({'user_email': user_email})
    list_query = list(query)
    mongo_bpm_df = pd.DataFrame({
        'ds': [list_query[x]['timestamp'] for x in range(len(list_query))],
        'bpm': [list_query[x]['value'] for x in range(len(list_query))]
    })
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')    
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    mongo_bpm_df = mongo_bpm_df.astype({'y': 'int32'})

    hour_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=180)]
    hour_df = hour_df.groupby(hour_df['ds'].dt.floor('h')).agg({'y':'mean'}).reset_index()
    
    hour_model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    hour_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    hour_model.add_country_holidays(country_name='KOR')
    hour_model.fit(hour_df)
    
    hour_future = hour_model.make_future_dataframe(periods=72, freq='h')
    hour_forecast = hour_model.predict(hour_future)
    
    hour_forecast.rename(columns={'yhat': 'hour_pred_bpm'}, inplace=True)
    hour_forecast['hour_pred_bpm'] = np.round(hour_forecast['hour_pred_bpm'], 3)

    return {'hour_pred_bpm': hour_forecast[['ds', 'hour_pred_bpm']][len(hour_forecast) - 72:].to_dict('records')}
    
    
@app.post("/check_db3")
async def check_db_query(request: UserEmailRequest):
    save_date = (datetime.now() + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')
    user_email = request.user_email
    record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
    collection_names = ['bpm', 'steps', 'calories', 'sleeps']

    # 서버 dynamoDB 데이터 load 및 DataFrame 생성 (1)
    dynamo_start_time = datetime.now()
    exist_times = exist_collection(user_email, collection_names)
    json_data = [new_query(user_email, record_names[x], exist_times[x]) for x in range(len(exist_times))]
    df_data = [create_df(json_data[x]) for x in range(len(json_data))]
    
    # df_data[0] : bpm dataframe
    print(df_data)
    dynamo_end_time = datetime.now()
    print(f'DynamoDB Data 불러오기 및 DataFrame 생성 : {dynamo_end_time - dynamo_start_time}s')
    
    # 서버 MongoDB 데이터 save (2)
    mongo_start_time = datetime.now()    
    [update_db(user_email, df_data[x], collection_names[x], save_date) for x in range(len(df_data))]
    mongo_end_time = datetime.now()
    print(f'MongoDB 저장 : {mongo_end_time - mongo_start_time}s')
    
    
@app.get("/feature_day/{user_email}")
async def bpm_day_feature(user_email: str):
    
    mongo_bpm_df = pd.DataFrame(bpm.find_one({'user_email': user_email})['data'])
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')
    mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

    day_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=730)]
    day_group = day_df.groupby(day_df['ds'].dt.floor('d'))
    day_hrv = day_group.apply(calc_hrv).reset_index()
    
    day_hrv.rename(columns={'rmssd' : 'day_rmssd', 'sdnn' : 'day_sdnn'}, inplace=True)
    print(day_hrv)
    print('------------------------------------------')
    
    return {'day_hrv': day_hrv[['ds', 'day_rmssd', 'day_sdnn']].to_dict('records')}    

@app.get("/feature_hour/{user_email}")
async def bpm_hour_feature(user_email: str):
    # 서버 HRV 계산 hour (6)
    hrv_hour_start_time = time.time()
    mongo_bpm_df = pd.DataFrame(bpm.find_one({'user_email': user_email})['data'])
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')
    mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

    hour_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=180)]
    hour_group = hour_df.groupby(hour_df['ds'].dt.floor('h'))
    hour_hrv = hour_group.apply(calc_hrv).reset_index()
    
    hour_hrv.rename(columns={'rmssd' : 'hour_rmssd', 'sdnn' : 'hour_sdnn'}, inplace=True)
    hrv_hour_end_time = time.time()
    print(f'HRV 계산 hour 걸린 시간 : {hrv_hour_end_time - hrv_hour_start_time}')

    return {'hour_hrv': hour_hrv[['ds', 'hour_rmssd', 'hour_sdnn']].to_dict('records')}

@app.get("/predict_minute/{user_email}")
async def bpm_minute_predict(user_email: str):
    # 서버 BPM Prediction minute (4)
    pred_min_start_time = time.time()
    mongo_bpm_df = pd.DataFrame(bpm.find_one({'user_email': user_email})['data'])
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')    
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    mongo_bpm_df = mongo_bpm_df.astype({'y': 'int32'})
    
    # if types == 'min':
    
    min_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=7)]
    
    
    
    start_time = time.time()
    min_model = Prophet(
        changepoint_prior_scale=0.0001,
        seasonality_mode='multiplicative',
        #daily_seasonality=True,
        #weekly_seasonality=True,
        #yearly_seasonality=True,
        #interval_width=0.95
    )
    min_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    min_model.add_country_holidays(country_name='KOR')
    min_model.fit(min_df)
    
    min_future = min_model.make_future_dataframe(periods=60*24*1, freq='min')
    min_forecast = min_model.predict(min_future)
    
    
    min_forecast.rename(columns={'yhat': 'min_pred_bpm'}, inplace=True)
    min_forecast['min_pred_bpm'] = np.round(min_forecast['min_pred_bpm'], 3)
    
    min_forecast = min_forecast[len(min_forecast) - 60*24*1:]
    
    pred_min_end_time = time.time()
    print(f'BPM predict minute 걸린 시간 : {pred_min_end_time - pred_min_start_time}')
        
    return {'min_pred_bpm': min_forecast[['ds', 'min_pred_bpm']].to_dict('records')}
    
    
    
@app.get("/predict_hour/{user_email}")
async def bpm_hour_predict(user_email: str):
    # 서버 BPM Prediction hour (5)
    pred_hour_start_time = time.time()
    mongo_bpm_df = pd.DataFrame(bpm.find_one({'user_email': user_email})['data'])
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')    
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    mongo_bpm_df = mongo_bpm_df.astype({'y': 'int32'})
    
    # if types == 'min':
    
    hour_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=180)]
    hour_df = hour_df.groupby(hour_df['ds'].dt.floor('h')).agg({'y':'mean'}).reset_index()
    
    hour_model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    hour_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    hour_model.add_country_holidays(country_name='KOR')
    hour_model.fit(hour_df)
    
    hour_future = hour_model.make_future_dataframe(periods=72, freq='h')
    hour_forecast = hour_model.predict(hour_future)
    
    hour_forecast.rename(columns={'yhat': 'hour_pred_bpm'}, inplace=True)
    hour_forecast['hour_pred_bpm'] = np.round(hour_forecast['hour_pred_bpm'], 3)
    
    pred_hour_end_time = time.time()
    print(f'BPM predict hour 걸린 시간 : {pred_hour_end_time - pred_hour_start_time}')
    
    
        
    return {'hour_pred_bpm': hour_forecast[['ds', 'hour_pred_bpm']][len(hour_forecast) - 72:].to_dict('records')}
        
        
@app.get("/predict_day/{user_email}")
async def bpm_day_predict(user_email: str):
    mongo_bpm_df = pd.DataFrame(bpm.find_one({'user_email': user_email})['data'])
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df['hour_rounded'] = mongo_bpm_df['ds'].dt.floor('h')
    mongo_bpm_df['day_rounded'] = mongo_bpm_df['ds'].dt.floor('d')    
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    mongo_bpm_df = mongo_bpm_df.astype({'y': 'int32'})
    
    # if types == 'min':
    
    day_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] - timedelta(days=730)]
    day_df = day_df.groupby(day_df['ds'].dt.floor('d')).agg({'y':'mean'}).reset_index()
    
    day_model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    day_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    day_model.add_country_holidays(country_name='KOR')
    day_model.fit(day_df)
    
    day_future = day_model.make_future_dataframe(periods=3, freq='d')
    day_forecast = day_model.predict(day_future)
    
    day_forecast.rename(columns={'yhat': 'day_pred_bpm'}, inplace=True)
    day_forecast['day_pred_bpm'] = np.round(day_forecast['day_pred_bpm'], 3)
        
    return {'day_pred_bpm': day_forecast[['ds', 'day_pred_bpm']][len(day_forecast) - 30:].to_dict('records')}
    
    
    # ############## 유저 데이터 X -> 전체 데이터 쿼리 ##############
    # if bpm_collection.find_one({'user_email': user_email}) == None or steps_collection.find_one({'user_email': user_email}) == None or calories_collection.find_one({'user_email': user_email}) == None or sleeps_collection.find_one({'user_email': user_email}) == None:
    #     start_time = time.time()
    #     if bpm_collection.find_one({'user_email': user_email}) == None:
    #         bpm_query = all_query(user_email, record_names[0])
            
    #         all_bpm_df = pd.DataFrame({
    #             'ds': [bpm_query[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(bpm_query))],
    #             'bpm': [int(bpm_query[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(bpm_query))],
    #         })
            
    #         all_bpm_df['ds'] = pd.to_datetime(all_bpm_df['ds'])
            
    #         bpm_collection.insert_one({
    #             'user_email': user_email,
    #             'save_date': request_time,
    #             'last_date': list(all_bpm_df['ds'])[-1],
    #             'data': all_bpm_df.to_dict('records'),
    #         })
            
            
            
            
    #     if steps_collection.find_one({'user_email': user_email}) == None:
    #         step_query = all_query(user_email, record_names[1])
            
    #         all_step_df = pd.DataFrame({
    #             'ds': [step_query[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(step_query))],
    #             'step': [int(step_query[x]['recordInfo']['M']['count']['N']) for x in range(len(step_query))],
    #         })
            
    #         all_step_df['ds'] = pd.to_datetime(all_step_df['ds'])
            
    #         steps_collection.insert_one({
    #             'user_email': user_email,
    #             'save_date': request_time,
    #             'last_date': list(all_step_df['ds'])[-1],
    #             'data': all_step_df.to_dict('records'),
    #         })
            
            
            
            
            
    #     if calories_collection.find_one({'user_email': user_email}) == None:
    #         calorie_query = all_query(user_email, record_names[2])
            
    #         all_calorie_df = pd.DataFrame({
    #             'ds': [calorie_query[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(calorie_query))],
    #             'calorie': [np.round(float(calorie_query[x]['recordInfo']['M']['energy']['M']['value']['N']), 3) for x in range(len(calorie_query))],
    #         })
            
    #         all_calorie_df['ds'] = pd.to_datetime(all_calorie_df['ds'])
            
    #         calories_collection.insert_one({
    #             'user_email': user_email,
    #             'save_date': request_time,
    #             'last_date': list(all_calorie_df['ds'])[-1],
    #             'data': all_calorie_df.to_dict('records'),
    #         })
            
            
            
            
    #     if sleeps_collection.find_one({'user_email': user_email}) == None:
    #         sleeps_query = all_query(user_email, record_names[3])
            
    #         all_sleep_df = pd.DataFrame({
    #             'ds_start': [sleeps_query[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(sleeps_query)) for y in range(len(sleeps_query[x]['recordInfo']['M']['stages']['L']))],
    #             'ds_end': [sleeps_query[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'].replace('T', ' ')[:19] for x in range(len(sleeps_query)) for y in range(len(sleeps_query[x]['recordInfo']['M']['stages']['L']))],
    #             'stage': [int(sleeps_query[x]['recordInfo']['M']['stages']['L'][y]['M']['stage']['N']) for x in range(len(sleeps_query)) for y in range(len(sleeps_query[x]['recordInfo']['M']['stages']['L']))],
    #         })
            
    #         all_sleep_df['ds_start'] = pd.to_datetime(all_sleep_df['ds_start'])
    #         all_sleep_df['ds_end'] = pd.to_datetime(all_sleep_df['ds_end'])
            
    #         sleeps_collection.insert_one({
    #             'user_email': user_email,
    #             'save_date': request_time,
    #             'last_date': pd.to_datetime(sleeps_query[-1]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19]),
    #             'data': all_sleep_df.to_dict('records'),
    #         })
            
            
            
    #     bpm_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #         },
    #         upsert=True
    #     )
        
    #     steps_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #         },
    #         upsert=True
    #     )

    #     calories_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #         },
    #         upsert=True
    #     )

    #     sleeps_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #         },
    #         upsert=True
    #     )
    #     end_time = time.time()
    #     print(f'데이터 저장 및 save 시간 업데이트 걸린 시간 : {(end_time - start_time):.4f}')
        
    
    # # else:
    # start_time = time.time()
    # mongo_bpm_last_ds = pd.to_datetime(bpm_collection.find_one({'user_email': user_email})['last_date'])
    # mongo_steps_last_ds = pd.to_datetime(steps_collection.find_one({'user_email': user_email})['last_date'])
    # mongo_calories_last_ds = pd.to_datetime(calories_collection.find_one({'user_email': user_email})['last_date'])
    # mongo_sleeps_last_ds = pd.to_datetime(sleeps_collection.find_one({'user_email': user_email})['last_date'])
    # end_time = time.time()
    # print(f'몽고디비 마지막 last date 가져오는 시간 (4개) : {(end_time - start_time):.4f}')
    
    # start_time = time.time()
    # query_bpm_last_ds = pd.to_datetime(one_query(user_email, record_names[0])['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19])
    # query_steps_last_ds = pd.to_datetime(one_query(user_email, record_names[1])['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19])
    # query_calories_last_ds = pd.to_datetime(one_query(user_email, record_names[2])['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19])
    # query_sleeps_last_ds = pd.to_datetime(one_query(user_email, record_names[3])['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19])
    # end_time = time.time()
    # print(f'마지막 쿼리 하나 가져오는데 걸리는 시간 (4개) : {(end_time - start_time):.4f}')
    
    # print(f'Mongo Last ds : {mongo_bpm_last_ds} <----> Query Last ds : {query_bpm_last_ds}')
    # print(f'Mongo Last ds : {mongo_steps_last_ds} <----> Query Last ds : {query_steps_last_ds}')
    # print(f'Mongo Last ds : {mongo_calories_last_ds} <----> Query Last ds : {query_calories_last_ds}')
    # print(f'Mongo Last ds : {mongo_sleeps_last_ds} <----> Query Last ds : {query_sleeps_last_ds}')
    # start_time = time.time()
    # if mongo_bpm_last_ds == query_bpm_last_ds:
    #     print('bpm 업데이트 데이터 없음')
    #     # pass
    # else:
    #     new_bpm = new_query(user_email, record_names[0], str(mongo_bpm_last_ds).replace(' ', 'T'))
    #     new_bpm_df = pd.DataFrame({
    #         'ds': [new_bpm[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(new_bpm))],
    #         'bpm': [int(new_bpm[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(new_bpm))],
    #     })
        
    #     # request_time
        
    #     bpm_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #             '$push': {
    #                 'data': {
    #                     '$each': new_bpm_df.to_dict('records')
    #                 }
    #             }
    #         },
    #         upsert=True
    #     )
    # end_time = time.time()
    # print(f'BPM 데이터 업데이트 체크 걸린 시간 : {(end_time - start_time):.4f}')
        
        
    # start_time = time.time()
    # if mongo_steps_last_ds == query_steps_last_ds:
    #     print('걸음수 업데이트 데이터 없음')
    #     # pass
    # else:
    #     new_steps = new_query(user_email, record_names[1], str(mongo_steps_last_ds).replace(' ', 'T'))
    #     new_steps_df = pd.DataFrame({
    #         'ds': [new_steps[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(new_steps))],
    #         'step': [int(new_steps[x]['recordInfo']['M']['count']['N']) for x in range(len(new_steps))],
    #     })
        
    #     # request_time
        
    #     steps_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #             '$push': {
    #                 'data': {
    #                     '$each': new_steps_df.to_dict('records')
    #                 }
    #             }
    #         },
    #         upsert=True
    #     )
        
    # end_time = time.time()
    # print(f'step 데이터 업데이트 체크 걸린 시간 : {(end_time - start_time):.4f}')
    
    # start_time = time.time()
    # if mongo_calories_last_ds == query_calories_last_ds:
    #     print('칼로리소모량 업데이트 데이터 없음')
    #     # pass
    # else:
    #     new_calories = new_query(user_email, record_names[2], str(mongo_calories_last_ds).replace(' ', 'T'))
    #     new_calories_df = pd.DataFrame({
    #         'ds': [new_calories[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(new_calories))],
    #         'calorie': [np.round(float(new_calories[x]['recordInfo']['M']['energy']['M']['value']['N']), 3) for x in range(len(new_calories))],
    #     })
        
    #     calories_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #             '$push': {
    #                 'data': {
    #                     '$each': new_calories_df.to_dict('records')
    #                 }
    #             }
    #         },
    #         upsert=True
    #     )    
    # end_time = time.time()
    # print(f'칼로리 데이터 업데이트 체크 걸린 시간 : {(end_time - start_time):.4f}')
    
    
    # start_time = time.time()
    # if mongo_sleeps_last_ds == query_sleeps_last_ds:
    #     print('수면 업데이트 데이터 없음')
    #     # pass
    # else:
    #     new_sleeps = new_query(user_email, record_names[3], str(mongo_sleeps_last_ds).replace(' ', 'T'))
    #     new_sleeps_df = pd.DataFrame({
    #         'ds_start': [new_sleeps[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(new_sleeps)) for y in range(len(new_sleeps[x]['recordInfo']['M']['stages']['L']))],
    #         'ds_end': [new_sleeps[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'].replace('T', ' ')[:19] for x in range(len(new_sleeps)) for y in range(len(new_sleeps[x]['recordInfo']['M']['stages']['L']))],
    #         'stage': [int(new_sleeps[x]['recordInfo']['M']['stages']['L'][y]['M']['stage']['N']) for x in range(len(new_sleeps)) for y in range(len(new_sleeps[x]['recordInfo']['M']['stages']['L']))],
    #     })
        
    #     # request_time
        
    #     sleeps_collection.update_one(
    #         {'user_email': user_email},
    #         {
    #             '$set': {
    #                 'save_date': request_time,
    #             },
    #             '$push': {
    #                 'data': {
    #                     '$each': new_sleeps_df.to_dict('records')
    #                 }
    #             }
    #         },
    #         upsert=True
    #     )
    # end_time = time.time()
    # print(f'수면 데이터 업데이트 체크 걸린 시간 : {(end_time - start_time):.4f}')



        
        
    # # elif types == 'hour':
        
    # hour_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] >= timedelta(days=180)]
    # hour_df = hour_df.groupby(hour_df['ds'].dt.floor('h')).agg({'y':'mean'}).reset_index()
    
    # hour_model = Prophet(
    #     changepoint_prior_scale=0.01,
    #     seasonality_mode='multiplicative',
    #     daily_seasonality=True,
    #     weekly_seasonality=True,
    #     yearly_seasonality=True,
    #     interval_width=0.95
    # )
    # hour_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    # hour_model.add_country_holidays(country_name='KOR')
    # hour_model.fit(hour_df)
    
    # hour_future = hour_model.make_future_dataframe(periods=10, freq='d')
    # hour_forecast = hour_model.predict(hour_future)
    
    # hour_forecast.rename(columns={'yhat': 'pred_bpm'}, inplace=True)
    # hour_forecast['pred_bpm'] = np.round(hour_forecast['pred_bpm'], 3)
        
    #     # return {'hour_pred_bpm': hour_forecast[['ds', 'hour_pred_bpm']].to_dict('records')}
    
    # # elif types == 'day':
    
    # day_df = mongo_bpm_df[mongo_bpm_df.day_rounded >= mongo_bpm_df.day_rounded[len(mongo_bpm_df) - 1] >= timedelta(days=730)]
    # day_df = day_df.groupby(day_df['ds'].dt.floor('d')).agg({'y':'mean'}).reset_index()
    
    # day_model = Prophet(
    #     changepoint_prior_scale=0.01,
    #     seasonality_mode='multiplicative',
    #     daily_seasonality=True,
    #     weekly_seasonality=True,
    #     yearly_seasonality=True,
    #     interval_width=0.95
    # )
    # day_model.add_seasonality(name='hourly', period=24, fourier_order=7)
    # day_model.add_country_holidays(country_name='KOR')
    # day_model.fit(day_df)
    
    # day_future = day_model.make_future_dataframe(periods=30, freq='d')
    # day_forecast = day_model.predict(day_future)
    
    # day_forecast.rename(columns={'yhat': 'pred_bpm'}, inplace=True)
    # day_forecast['pred_bpm'] = np.round(day_forecast['pred_bpm'], 3)
        
    #     # return {'day_pred_bpm': day_forecast[['ds', 'day_pred_bpm']].to_dict('records')}

    # return {
    #     'min_pred_bpm': min_forecast[['ds', 'pred_bpm']].to_dict('records'), 
    #     'hour_pred_bpm': hour_forecast[['ds', 'pred_bpm']].to_dict('records'), 
    #     'day_pred_bpm': day_forecast[['ds', 'pred_bpm']].to_dict('records'),
    # }
        




def query_bpm_data(user_email: str):
    bpm_items = []
    last_evaluated_key = None
    
    try:
        while True:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'HeartRateRecord#'},
                },
                'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            bpm_items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        # 원래 순서로 되돌리기
        bpm_items.reverse()
        
        return bpm_items
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
    
def query_step_data(user_email: str):
    step_items = []
    last_evaluated_key = None
    
    try:
        while True:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'StepsRecord#'},
                },
                'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            step_items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        # 원래 순서로 되돌리기
        step_items.reverse()
        
        return step_items
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
    
def query_calorie_data(user_email: str):
    calorie_items = []
    last_evaluated_key = None
    
    try:
        while True:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'TotalCaloriesBurnedRecord#'},
                },
                'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            calorie_items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
        
        # 원래 순서로 되돌리기
        calorie_items.reverse()
        
        return calorie_items
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
       
def query_one_data(user_email: str):
    try:
        query_params = {
            'TableName': TABLE_NAME,
            'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': f'U#{user_email}'},
                ':sk_prefix': {'S': f'HeartRateRecord#'},
            },
            'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            'Limit': 1  # 딱 1개의 항목만 요청
        }
        
        response = dynamodb.query(**query_params)
        
        if response['Items']:
            return response['Items'][0]  # 첫 번째(가장 최신) 항목만 반환
        else:
            return None  # 결과가 없을 경우
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None

# def calc_hrv(group):
#     rr_intervals = 60000 / group['bpm'].values
#     peaks = nk.intervals_to_peaks(rr_intervals)
#     hrv = nk.hrv_time(peaks)
#     return pd.Series({
#         'rmssd': hrv['HRV_RMSSD'].values[0],
#         'sdnn': hrv['HRV_SDNN'].values[0],
#     })

def create_bpm_dataframe_(bpm_data):
    if not bpm_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    
    df = pd.DataFrame({
        'ds': [bpm_data[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(bpm_data))],
        'bpm': [int(bpm_data[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(bpm_data))]
    })
    
    df['ds'] = pd.to_datetime(df['ds'])
    
    last_ds = list(df['ds'])[-1]
    
    grouped_hour = df.groupby(df['ds'].dt.floor('h'))
    bpm_hour = np.round(grouped_hour['bpm'].mean(), 3)
    hrv_hour = grouped_hour.apply(calc_hrv)
    
    grouped_day = df.groupby(df['ds'].dt.floor('d'))
    bpm_day = np.round(grouped_day['bpm'].mean(), 3)
    hrv_day = grouped_day.apply(calc_hrv)
    
    # 매 시간으로 BPM 평균..? 
    hour_df = pd.DataFrame({
        'ds': bpm_hour.index,
        'bpm': bpm_hour.values,
        'rmssd': hrv_hour['rmssd'].values,
        'sdnn': hrv_hour['sdnn'].values,
    })
    
    day_df = pd.DataFrame({
        'ds': bpm_day.index,
        'bpm': bpm_day.values,
        'rmssd': hrv_day['rmssd'].values,
        'sdnn': hrv_day['sdnn'].values,
    })
    
    
    ############ BPM Hour Prediction ##########
    pred_hour_df = hour_df.rename(columns={'bpm':'y'})
    
    model_hbpm = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    model_hbpm.add_seasonality(name='hourly', period=24, fourier_order=7)
    model_hbpm.add_country_holidays(country_name='KOR')
    model_hbpm.fit(pred_hour_df[:])
    
    future_hbpm = model_hbpm.make_future_dataframe(periods=24*3, freq='h')
    forecast_hbpm = model_hbpm.predict(future_hbpm)
    
    ds_df = pd.DataFrame({'ds': [forecast_hbpm['ds'][x] for x in range(len(forecast_hbpm))]})
    new_hour_df = pd.merge(ds_df, hour_df[['ds','bpm','rmssd','sdnn']], on='ds', how='left')
    new_hour_df = pd.merge(new_hour_df, forecast_hbpm[['ds','yhat']], on='ds', how='left')
    new_hour_df = new_hour_df.rename(columns={'yhat':'pred_bpm'})
    
    pred_hour_df = hour_df.rename(columns={'y':'bpm'})
    ############## BPM Hour Prediction End ##############
    
    ############## RMSSD Hour Prediction #################
    pred_hour_df = hour_df.rename(columns={'rmssd':'y'})
    
    model_hrmssd = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    model_hrmssd.add_seasonality(name='hourly', period=24, fourier_order=7)
    model_hrmssd.add_country_holidays(country_name='KOR')
    model_hrmssd.fit(pred_hour_df[:])
    
    future_hrmssd = model_hrmssd.make_future_dataframe(periods=24*3, freq='h')
    forecast_hrmssd = model_hrmssd.predict(future_hrmssd)
    

    new_hour_df = pd.merge(new_hour_df, forecast_hrmssd[['ds','yhat']], on='ds', how='left')
    new_hour_df = new_hour_df.rename(columns={'yhat':'pred_rmssd'})
    ################# RMSSD Hour Prediction End ############
    
    ################# BPM Day Prediction ############ (임시로...)
    pred_day_df = day_df.rename(columns={'bpm':'y'})
    
    model_dbpm = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    model_dbpm.add_seasonality(name='monthly', period=30.5, fourier_order=7) # 수정 필요 가능성
    model_dbpm.add_country_holidays(country_name='KOR')
    model_dbpm.fit(pred_day_df[:])
    
    future_dbpm = model_dbpm.make_future_dataframe(periods=5, freq='d') # 5일 예측
    forecast_dbpm = model_dbpm.predict(future_dbpm)
    
    ds_df = pd.DataFrame({'ds': [forecast_dbpm['ds'][x] for x in range(len(forecast_dbpm))]})
    new_day_df = pd.merge(ds_df, day_df[['ds','bpm','rmssd','sdnn']], on='ds', how='left')
    new_day_df = pd.merge(new_day_df, forecast_dbpm[['ds','yhat']], on='ds', how='left')
    new_day_df = new_day_df.rename(columns={'yhat':'pred_bpm'})
    
    pred_day_df = hour_df.rename(columns={'y':'bpm'})
    ################# BPM Day Prediction End ###########
    
    ################# BPM Day Prediction ############ (임시로...)
    pred_day_df = day_df.rename(columns={'rmssd':'y'})
    
    model_drmssd = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    model_drmssd.add_seasonality(name='monthly', period=30.5, fourier_order=7) # 수정 필요 가능성
    model_drmssd.add_country_holidays(country_name='KOR')
    model_drmssd.fit(pred_day_df[:])
    
    future_drmssd = model_drmssd.make_future_dataframe(periods=5, freq='d') # 5일 예측
    forecast_drmssd = model_drmssd.predict(future_drmssd)
    
    new_day_df = pd.merge(new_day_df, forecast_drmssd[['ds','yhat']], on='ds', how='left')
    new_day_df = new_day_df.rename(columns={'yhat':'pred_rmssd'})

    ################# BPM Day Prediction End ###########
    
    return new_hour_df, new_day_df, last_ds
    
def create_step_dataframe_(step_data):
    if not step_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    
    df = pd.DataFrame({
        'ds': [step_data[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(step_data))],
        'step': [int(step_data[x]['recordInfo']['M']['count']['N']) for x in range(len(step_data))],
    })
    
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 초 단위가 0이 아닌 경우들 제외
    df['second'] = df['ds'].dt.second
    df = df[df.second == 0]
    
    hour_df = df.groupby(df['ds'].dt.floor('h')).agg({
        'step': 'sum',
    }).reset_index()
    
    day_df = df.groupby(df['ds'].dt.floor('d')).agg({
        'step': 'sum',
    }).reset_index()
    
    return hour_df, day_df

def create_calorie_dataframe_(calorie_data):
    if not calorie_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    
    df = pd.DataFrame({
        'ds': [calorie_data[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(calorie_data))],
        'calorie': [np.round(float(calorie_data[x]['recordInfo']['M']['energy']['M']['value']['N'])) for x in range(len(calorie_data))],
    })
    
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 초 단위가 0이 아닌 경우들 제외
    df['second'] = df['ds'].dt.second
    df = df[df.second == 0]
    
    hour_df = df.groupby(df['ds'].dt.floor('h')).agg({
        'calorie': 'sum',
    }).reset_index()
    
    day_df = df.groupby(df['ds'].dt.floor('d')).agg({
        'calorie': 'sum',
    }).reset_index()
    
    return hour_df, day_df

################################


# 시간을 한국 시간으로 변형 (칼로리, 심박수)
def conv_ds(startTime):
    startTime = startTime.replace('T', ' ')
    startTime = startTime.replace('Z', '')
    return add_9(startTime)

def add_9(startTime):
    Year = startTime[:4]
    Month = startTime[5:7]
    Day = startTime[8:10]
    Hour = startTime[11:13]
    Min = startTime[14:16]
    Sec = startTime[17:19]
    if (int(Hour) + 9) >= 24 :
        if get_month_lastday(int(Year), int(Month)) == int(Day):
            if int(Month) == 12:
                Year = str(int(Year) + 1)
                Month = '01'
                Day = '01'
                Hour = str(int(Hour) + 9 - 24).zfill(2)
            else:
                Month = str(int(Month) + 1).zfill(2)
                Day = '01'
                Hour = str(int(Hour) + 9 - 24).zfill(2)
        else:
            Day = str(int(Day) + 1).zfill(2)
            Hour = str(int(Hour) + 9 - 24).zfill(2)
    else:
        Hour = str(int(Hour) + 9).zfill(2)     
    return Year + '-' + Month + '-' + Day + ' ' + Hour + ':' + Min + ':' + Sec
        
def get_month_lastday(year, month):
    next_month = datetime(year=year, month=month, day=1).date() + relativedelta.relativedelta(months=1)
    return (next_month - timedelta(days=1)).day

# 걸음수 데이터 시간 처리
def conv_ds_step(time):
    time = time.replace('T', ' ')
    return time

# 수면 데이터 시간 처리
def conv_ds_sleep(time):
    time = time.replace('T', ' ')
    time = time.replace('Z', '')
    return time[:-5]


############################# 칼로리 ###################################
# 마지막 데이터로부터 3000개의 데이터만 query -> calorie
def query_latest_calorie_data(user_email: str, limit: int = 3000):
    items = []
    last_evaluated_key = None
    try:
        while len(items) < limit:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'TotalCaloriesBurnedRecord#'},
                },
                'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
                'Limit': min(limit - len(items), 1000)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key or len(items) >= limit:
                break
        
        # 원래 순서로 되돌리기
        items.reverse()
        
        return items[:limit] 
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
    
# 마지막 데이터 1개만 query calorie
def query_one_calorie_data(user_email: str):
    try:
        query_params = {
            'TableName': TABLE_NAME,
            'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': f'U#{user_email}'},
                ':sk_prefix': {'S': f'TotalCaloriesBurnedRecord#'},
            },
            'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            'Limit': 1  # 딱 1개의 항목만 요청
        }
        
        response = dynamodb.query(**query_params)
        
        if response['Items']:
            return response['Items'][0]  # 첫 번째(가장 최신) 항목만 반환
        else:
            return None  # 결과가 없을 경우
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
############################# 칼로리 ###################################


############################# 심박수 ###################################
# 마지막 데이터로부터 40000개의 데이터만 query -> HeartRate
def query_latest_heart_rate_data(user_email: str, limit: int = 40000):
    items = []
    last_evaluated_key = None
    try:
        while len(items) < limit:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'HeartRateRecord#'},
                },
                'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
                'Limit': min(limit - len(items), 1000)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key or len(items) >= limit:
                break
        
        # 원래 순서로 되돌리기
        items.reverse()
        
        return items[:limit] 
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
    
# 마지막 데이터 1개만 query -> HeartRate (DynamoDB 데이터가 새로 동기화가 되었는지 확인 -> MongoDB에 저장된 데이터와 비교를 위해)    
def query_one_heart_rate_data(user_email: str):
    try:
        query_params = {
            'TableName': TABLE_NAME,
            'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': f'U#{user_email}'},
                ':sk_prefix': {'S': f'HeartRateRecord#'},
            },
            'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            'Limit': 1  # 딱 1개의 항목만 요청
        }
        
        response = dynamodb.query(**query_params)
        
        if response['Items']:
            return response['Items'][0]  # 첫 번째(가장 최신) 항목만 반환
        else:
            return None  # 결과가 없을 경우
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
############################# 심박수 ###################################


############################# 걸음수 ###################################
# 마지막 데이터로부터 40000개 데이터만 query (걸음수 데이터)
def query_latest_step_data(user_email: str, limit: int = 40000):
    items = []
    last_evaluated_key = None
    try:
        while len(items) < limit:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': 'StepsRecord#'},
                },
                'ScanIndexForward': False,
                'Limit': min(limit - len(items), 1000)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key or len(items) >= limit:
                break
        items.reverse()
        
        return items[:limit] 
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None

# 마지막 데이터 1개만 query (DynamoDB 데이터가 새로 동기화가 되었는지 확인 -> MongoDB에 저장된 데이터와 비교를 위해)  (걸음수 데이터)  
def query_one_step_data(user_email: str):
    try:
        query_params = {
            'TableName': TABLE_NAME,
            'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': f'U#{user_email}'},
                ':sk_prefix': {'S': f'StepsRecord#'},
            },
            'ScanIndexForward': False,  
            'Limit': 1 
        }
        
        response = dynamodb.query(**query_params)
        
        if response['Items']:
            return response['Items'][0]  
        else:
            return None  
    except ClientError as e:
        print(f"an error occurred: {e.response['Error']['Message']}")
        return None
############################# 걸음수 ###################################


############################# 수면 ###################################
# 마지막 데이터로부터 60개(2달치) 데이터만 query (수면 데이터)
def query_latest_sleep_data(user_email: str, limit: int = 60):
    items = []
    last_evaluated_key = None
    try:
        while len(items) < limit:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':sk_prefix': {'S': f'SleepSessionRecord#'},
                },
                'ScanIndexForward': False,
                'Limit': min(limit - len(items), 1000)
            }
            
            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key
            
            response = dynamodb.query(**query_params)
            
            items.extend(response['Items'])
            
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key or len(items) >= limit:
                break
        items.reverse()
        
        return items[:limit] 
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None

# 마지막 데이터 1개만 query (DynamoDB 데이터가 새로 동기화가 되었는지 확인 -> MongoDB에 저장된 데이터와 비교를 위해)  (수면 데이터)  
def query_one_sleep_data(user_email: str):
    try:
        query_params = {
            'TableName': TABLE_NAME,
            'KeyConditionExpression': 'PK = :pk AND begins_with(SK, :sk_prefix)',
            'ExpressionAttributeValues': {
                ':pk': {'S': f'U#{user_email}'},
                ':sk_prefix': {'S': f'SleepSessionRecord#'},
            },
            'ScanIndexForward': False,  # 역순으로 정렬 (최신 데이터부터)
            'Limit': 1  # 딱 1개의 항목만 요청
        }
        
        response = dynamodb.query(**query_params)
        
        if response['Items']:
            return response['Items'][0]  # 첫 번째(가장 최신) 항목만 반환
        else:
            return None  # 결과가 없을 경우
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None
############################# 수면 ###################################


def create_calorie_dataframe(json_data):
    if not json_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    df = process_calorie_data(json_data)
    
    return df


def create_heart_rate_dataframe(json_data):
    if not json_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    processed_load_data = process_heart_rate_data(json_data)
    if not processed_load_data:
        raise HTTPException(status_code=404, detail="유저의 데이터가 없음")
    
    df = pd.DataFrame(processed_load_data)
    df['ds'] = pd.to_datetime(df['ds'])
    
    return df

def create_step_dataframe(json_data):
    if not json_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    df = process_step_data(json_data)
    
    return df

def create_sleep_dataframe(json_data):
    if not json_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    df = process_sleep_data(json_data)
    # if not df:
    #     raise HTTPException(status_code=404, detail="유저의 데이터가 없음")
    
    
    df['ds_start'] = pd.to_datetime(df['ds_start'])
    df['ds_end'] = pd.to_datetime(df['ds_end'])

    return df 


def process_calorie_data(items):
    ds = []
    calorie = []
    
    for i in range(len(items)):
        ds.append(conv_ds(items[i]['startTime']['S']))
        calorie.append(np.round(float(items[i]['recordInfo']['M']['energy']['M']['value']['N']),3))
        
    df = pd.DataFrame({
        'ds': ds,
        'calorie': calorie
    })
    
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    df = df.resample('h').sum()
    df = df.reset_index()
    
    return df

def process_heart_rate_data(items):
    processed_data = []
    for item in items:
        sk_parts = item['SK']['S'].split('#')
        if len(sk_parts) >= 2 and sk_parts[0] == 'HeartRateRecord':
            timestamp = sk_parts[1]
            converted_timestamp = conv_ds(timestamp)
            processed_data.append({
                'ds': converted_timestamp,
                'y': int(item['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N'])
            })
    return processed_data

def process_step_data(items):
    starttime = []
    step = []
    
    for i in range(len(items)):
        starttime.append(conv_ds_step(items[i]['startTime']['S'][:16]) + ':00')
        step.append(int(items[i]['recordInfo']['M']['count']['N']))
        
    df = pd.DataFrame({
        'ds': starttime,
        'step': step
    })
    
    df['ds'] = pd.to_datetime(df['ds']) # 에러
    
    df = df.set_index('ds')
    df = df.resample('h').sum()
    df = df.reset_index()
    
    return df



# 바로 수면 데이터의 데이터프레임 만들기
def process_sleep_data(items):
    starttime = []
    endtime = []
    stage = []

    for i in range(len(items)):
        for j in range(len(items[i]['recordInfo']['M']['stages']['L'])):
            starttime.append(conv_ds_sleep(items[i]['recordInfo']['M']['stages']['L'][j]['M']['startTime']['S']))
            endtime.append(conv_ds_sleep(items[i]['recordInfo']['M']['stages']['L'][j]['M']['endTime']['S']))
            stage.append(items[i]['recordInfo']['M']['stages']['L'][j]['M']['stage']['N'])    
              
    df = pd.DataFrame({
        'ds_start' : starttime,
        'ds_end' : endtime,
        'stage' : stage
    })
    
    return df

def save_calorie_to_mongodb(user_email: str, calorie_data, input_date):
    korea_time = input_date
    
    calorie_collection.insert_one({
        'user_email': user_email,
        'calorie_date': str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        'data': calorie_data.to_dict('records')
    })


def save_prediction_to_mongodb(user_email: str, prediction_data, input_date):
    korea_time = input_date
    
    def clean_value(v):
        if pd.isna(v) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        if isinstance(v, float):
            return round(v, 1)
        return v

    # DataFrame의 모든 열에 clean_value 함수 적용
    data_dict = prediction_data.applymap(clean_value).to_dict('records')
    
    # 최종 검사
    for item in data_dict:
        if 'yhat' in item and item['yhat'] is not None:
            item['yhat'] = float(item['yhat'])
        if 'y' in item:
            if item['y'] is not None and not pd.isna(item['y']):
                item['y'] = int(item['y'])
            else:
                item['y'] = None
    
    prediction_collection.insert_one({
        "user_email": user_email,
        "prediction_date": str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        "data": data_dict
    })

def save_analysis_to_mongodb(user_email: str, analysis_data, input_date):
    korea_time = input_date
    
    sdnn_rmssd = analysis_data.to_dict('records')
    
    # Clean the data before saving
    cleaned_data = []
    for item in sdnn_rmssd:
        cleaned_item = {
            'ds': item['ds'],
            'sdnn': clean_value(item['sdnn']),
            'rmssd': clean_value(item['rmssd'])
        }
        cleaned_data.append(cleaned_item)
    
    analysis_collection.insert_one({
        "user_email": user_email,
        "analysis_date": str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        "data": cleaned_data
    })
    
def save_step_to_mongodb(user_email: str, step_data, input_date):
    korea_time = input_date
    
    step_collection.insert_one({
        'user_email': user_email,
        'step_date': str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        'data': step_data.to_dict('records')
    })

def save_sleep_to_mongodb(user_email: str, sleep_data, input_date):
    korea_time = input_date
    
    sleep_collection.insert_one({
        "user_email": user_email,
        "sleep_date": str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        "data": sleep_data.to_dict('records')
    })
    
def update_db(user_email, df, collection, save_date):
    if len(df) == 0:
        return 0
      
    eval(collection).update_one(
        {'user_email': user_email},
        {
            '$set': {
                'save_date': save_date
            },
            '$push': {
                'data': {
                    '$each': df.to_dict('records')
                }
            }
    }, upsert = True)
    
def exist_collection(user_email, collections):
    res_idx = []
    for idx in collections:
        if eval(idx).find_one({'user_email': user_email}) == None:
            res_idx.append('0000-00-00T00:00:00')
        else:
            if idx == 'sleeps':
                res_idx.append(str(eval(idx).find_one({'user_email': user_email})['data'][-1]['ds_end']).replace(' ', 'T'))
            else:
                res_idx.append(str(eval(idx).find_one({'user_email': user_email})['data'][-1]['ds']).replace(' ', 'T'))
    return res_idx

def create_df(query_json):
    if len(query_json) == 0:
        return []

    if 'HeartRate' in query_json[0]['SK']['S']:
        return pd.DataFrame({
                    'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
                    'bpm': [int(query_json[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(query_json))],
                })

    if 'Steps' in query_json[0]['SK']['S']:
        return pd.DataFrame({
                    'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
                    'step': [int(query_json[x]['recordInfo']['M']['count']['N']) for x in range(len(query_json))],
                })

    if 'TotalCaloriesBurned' in query_json[0]['SK']['S']:
        return pd.DataFrame({
                    'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
                    'calorie': [np.round(float(query_json[x]['recordInfo']['M']['energy']['M']['value']['N']), 3) for x in range(len(query_json))],
                })

    if 'SleepSession' in query_json[0]['SK']['S']:
        return pd.DataFrame({
                    'ds_start': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
                    'ds_end': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
                    'stage': [int(query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['stage']['N']) for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))],
                })
        
def new_query(user_email: str, record_name: str, start_time: str):
    new_items = []
    start_time = start_time
    last_evaluated_key = None

    if record_name == 'Steps':
        start_time = start_time
    else:
        if str(start_time)[:4] == '0000':
            start_time = start_time
        else:
            if record_name == 'HeartRate' or record_name == 'TotalCaloriesBurned':
                start_time = str(pd.to_datetime(start_time) - timedelta(hours=9) + timedelta(minutes=1)).replace(' ', 'T')
            else:
                start_time = str(pd.to_datetime(start_time) - timedelta(hours=9)).replace(' ', 'T')
    try:
        while True:
            query_params = {
                'TableName': TABLE_NAME,
                'KeyConditionExpression': 'PK = :pk AND SK BETWEEN :start_sk AND :end_sk',
                'ExpressionAttributeValues': {
                    ':pk': {'S': f'U#{user_email}'},
                    ':start_sk': {'S':f'{record_name}Record#{start_time}'},
                    ':end_sk': {'S': f'{record_name}Record#9999-12-31T23:59:59Z'},
                },
                'ScanIndexForward': True,
            }

            if last_evaluated_key:
                query_params['ExclusiveStartKey'] = last_evaluated_key

            response = dynamodb.query(**query_params)
            new_items.extend(response['Items'])
            last_evaluated_key = response.get('LastEvaluatedKey')
            if not last_evaluated_key:
                break
            
        return new_items
            
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        return None

def predict_heart_rate(df):
    model = Prophet(changepoint_prior_scale=0.001,
                    changepoint_range=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    interval_width=0.95)
    model.fit(df)
    future = model.make_future_dataframe(periods=60*24*3, freq='min')
    forecast = model.predict(future)

    df['ds'] = pd.to_datetime(df['ds'])
    
    concat_df = forecast[['ds', 'yhat']]
    concat_df = concat_df.merge(df[['ds', 'y']], on='ds', how='left')
    concat_df = concat_df.where(pd.notnull(concat_df), None)
    return concat_df


@app.get("/calorie_dates/{user_email}")
async def get_calorie_dates(user_email: str):
    dates = calorie_collection.distinct('calorie_date', {'user_email': user_email})
    return {'dates': [date for date in dates]}

@app.get("/calorie_data/{user_email}/{calorie_date}")
async def get_calorie_data(user_email: str, calorie_date: str):
    try:
        calorie = calorie_collection.find_one({"user_email": user_email, "calorie_date": calorie_date})
        if calorie:
            calorie_data = []
            for item in calorie['data']:
                calorie_item = {
                    'ds': item['ds'],
                    'calorie': item['calorie']
                }
                calorie_data.append(calorie_item)
            return {"data": calorie_data}
        else:
            raise HTTPException(status_code=404, detail='calorie data not found')
    except ValueError:
        raise HTTPException(status_code=400, detail='invalid date format')


@app.get("/prediction_dates/{user_email}")
async def get_prediction_dates(user_email: str):
    dates = prediction_collection.distinct("prediction_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}

@app.get("/prediction_data/{user_email}/{prediction_date}")
async def get_prediction_data(user_email: str, prediction_date: str):
    try:
        prediction = prediction_collection.find_one({"user_email": user_email, "prediction_date": prediction_date})
        if prediction:
            def convert_types(item):
                return {
                    'ds': item['ds'],
                    'yhat': float(item['yhat']) if item['yhat'] is not None else None,
                    'y': int(item['y']) if item['y'] is not None else None
                }
            
            converted_data = [convert_types(item) for item in prediction["data"]]
            return {"data": converted_data}
        else:
            raise HTTPException(status_code=404, detail="Prediction data not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

@app.get("/prediction_data2/{user_email}/{prediction_date}")
async def get_prediction_data2(user_email: str, prediction_date: str):
    try:
        prediction = prediction_collection.find_one({"user_email": user_email, "prediction_date": prediction_date})
        if prediction:
            def convert_types(item):
                return {
                    'ds': item['ds'],
                    'yhat': float(item['yhat']) if item['yhat'] is not None else None,
                    'y': int(item['y']) if item['y'] is not None else None
                }
            
            converted_data = [convert_types(item) for item in prediction["data"]]
            #print(converted_data)
            return {"data": converted_data}
        else:
            raise HTTPException(status_code=404, detail="Prediction data not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

@app.get("/check_dates/{user_email}")
async def get_dates(user_email: str):
    dates = hourly_collection.distinct("date", {"user_email": user_email})
    return {"dates": [date for date in dates]}


@app.get("/analysis_dates/{user_email}")
async def get_analysis_dates(user_email: str):
    dates = analysis_collection.distinct("analysis_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}



@app.get("/analysis_data/{user_email}/{analysis_date}")
async def get_analysis_data(user_email: str, analysis_date: str):
    try:
        analysis = analysis_collection.find_one({"user_email": user_email, "analysis_date": analysis_date})
        if analysis:
            # Clean the data before sending
            cleaned_data = []
            for item in analysis['data']:
                cleaned_item = {
                    'ds': item['ds'],
                    'sdnn': clean_value(item['sdnn']),
                    'rmssd': clean_value(item['rmssd'])
                }
                cleaned_data.append(cleaned_item)
            return {"data": cleaned_data}
        else:

            raise HTTPException(status_code=404, detail="Analysis data not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")


@app.get("/step_dates/{user_email}")
async def get_step_dates(user_email: str):
    dates = step_collection.distinct("step_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}

@app.get("/step_data/{user_email}/{step_date}")
async def get_step_data(user_email: str, step_date: str):
    try:
        step = step_collection.find_one({"user_email": user_email, "step_date": step_date})
        if step:
            step_data = []
            for item in step['data']:
                step_item = {
                    'ds': item['ds'],
                    'step': item['step']
                }
                step_data.append(step_item)
            return {"data": step_data}
        else:
            raise HTTPException(status_code=404, detail="step data not fount")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")


@app.get("/sleep_dates/{user_email}")
async def get_sleep_dates(user_email: str):
    dates = sleep_collection.distinct("sleep_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}

@app.get("/sleep_data/{user_email}/{sleep_date}")
async def get_sleep_data(user_email: str, sleep_date: str):
    try:
        sleep = sleep_collection.find_one({"user_email": user_email, "sleep_date": sleep_date})
        if sleep:
            cleaned_data = []
            for item in sleep['data']:
                cleaned_item = {
                    'ds_start': item['ds_start'],
                    'ds_end': item['ds_end'],
                    'stage': clean_value(item['stage'])
                }
                cleaned_data.append(cleaned_item)
            return {"data": cleaned_data}
        else:
            raise HTTPException(status_code=404, detail="Sleep data not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")


def get_time_domain_features(nn_intervals) -> dict:
    # nn_intervals(ms) => 60000/bpm

    # nn intervals의 차이 (각 인접한 다음 nn interval의 차)
    diff_nni = np.diff(nn_intervals)
    # 데이터 길이
    length_int = len(nn_intervals)

    # Basic statistics
    # 
    mean_nni = np.mean(nn_intervals)
    median_nni = np.median(nn_intervals)
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int
    rmssd = np.std(np.diff(nn_intervals), ddof=1)

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features, heart_rate => bpm
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'rmssd': rmssd,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_nni': median_nni,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }

    return time_domain_features

def clean_value(v):
    if pd.isna(v) or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, float):
        return round(v, 1)
    return v

def preprocess_analysis(df):
    # 1시간 단위로 데이터 변경
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['hour'] = df['ds'].dt.hour
    df['minute'] = df['ds'].dt.minute
    
    df['ds_rounded'] = df['ds'].dt.floor('h')
    
    def divide_by_60000(y_list):
        return [np.round((60000 / y), 1) for y in y_list if y != 0]  # Avoid division by zero
    
    dict_temp = df.groupby('ds_rounded')['y'].apply(lambda x: divide_by_60000(x.tolist())).to_dict()

    # 키를 문자열 형식으로 변환
    dict_temp = {str(k): v for k, v in dict_temp.items()}
    
    key_list = []
    value_list = []
    
    for key in dict_temp.keys():
        key_list.append(key)
        
    for value in dict_temp.values():
        value_list.append(value)
        
    res_dict = {}
    sdnn = []
    rmssd = []
    
    for i in range(len(value_list)):
        if len(value_list[i]) > 1:  # Ensure there's enough data to calculate SDNN and RMSSD
            get_domain_hrv = get_time_domain_features(value_list[i])
            sdnn.append(clean_value(get_domain_hrv['sdnn']))
            rmssd.append(clean_value(get_domain_hrv['rmssd']))
        else:
            sdnn.append(None)
            rmssd.append(None)
    
    res_dict['ds'] = key_list
    res_dict['sdnn'] = sdnn
    res_dict['rmssd'] = rmssd
    
    analysis_df = pd.DataFrame(res_dict)
    analysis_df['ds'] = pd.to_datetime(analysis_df['ds'])
    
    return analysis_df

    
@app.post("/check_db2")
async def check_db(request: UserEmailRequest):
    user_email = request.user_email
    
    input_date = datetime.now() + timedelta(hours=9)
    
    
    if hourly_collection.find_one({'user_email': user_email}) == None: 
        
        bpm_data = query_bpm_data(user_email)
        step_data = query_step_data(user_email)
        calorie_data = query_calorie_data(user_email)
        
           
        bpm_hour, bpm_day, last_ds = create_bpm_dataframe_(bpm_data)
        step_hour, step_day = create_step_dataframe_(step_data)
        calorie_hour, calorie_day = create_calorie_dataframe_(calorie_data)
        
        hour_df = pd.concat([bpm_hour, step_hour, calorie_hour], axis=0).sort_values('ds').reset_index(drop=True).groupby('ds', as_index=False).first()
        day_df = pd.concat([bpm_day, step_day, calorie_day], axis=0).sort_values('ds').reset_index(drop=True).groupby('ds', as_index=False).first()
        
        hour_df = hour_df.replace({np.nan: None})
        day_df = day_df.replace({np.nan: None})
        
        hourly_collection.insert_one({
            'user_email': user_email,
            'date': str(input_date.year) + '-' + str(input_date.month).zfill(2) + '-' + str(input_date.day).zfill(2) + ' ' + str(input_date.hour).zfill(2) + ':' + str(input_date.minute).zfill(2) + ':' + str(input_date.second).zfill(2),
            'data': hour_df.to_dict('records'),
        })
        
        daily_collection.insert_one({
            'user_email': user_email,
            'date': str(input_date.year) + '-' + str(input_date.month).zfill(2) + '-' + str(input_date.day).zfill(2) + ' ' + str(input_date.hour).zfill(2) + ':' + str(input_date.minute).zfill(2) + ':' + str(input_date.second).zfill(2),
            'data': day_df.to_dict('records'),
        })
        
        return {'message': '데이터 저장 완료'}
    
    
    # collection 최신 데이터 반환
    query_data = list(db.hourly.aggregate([
        {'$match': {'user_email': user_email}},
        {'$sort': {'date': -1}},
        {'$limit': 1},
        {'$project': {
            'latest_date': '$date',
            'last_data_point': {'$arrayElemAt': ['$data', -1]}
        }}
    ]))
    
    
    
    # dict_compare = {'bpm': 'HeartRateRecord', 'rmssd': 'HeartRateRecord', 'sdnn': 'HeartRateRecord', 'step': 'StepsRecord', 'calorie': 'TotalCaloriesBurnedRecord'}
    
    
    
    
    # latest_doc = query_data[0]
    # print('latest_doc', latest_doc)
    
    # latest_key = list(latest_doc['last_data_point'].keys())[1:]
    # print('latest_key', latest_key)
    
    # latest_value = list(latest_doc['last_data_point'].values())[1:]
    # print('latest_value', latest_value)
    
    # latest_data_idx = [latest_value[x] == None for x in range(len(latest_value))].index(False)
    # print('latest_data_idx', latest_data_idx)
    
    # latest_date = pd.to_datetime(latest_doc['last_data_point']['ds'])
    # print('latest_date', latest_date)
    
    mongo_last_ds = query_data[0]['last_data_point']['ds']
    # query_sub72 = query_last_ds - timedelta(hours=72)
    
    # collection 최신 데이터의 마지막 데이터 중 None값이 아닌 첫 번째 데이터 index를 통해 query_one_data로 비교군 체크
    # record_name = dict_compare[latest_key[latest_data_idx]]
        
        
    # dict_compare = {'bpm': 'HeartRateRecord', 'rmssd': 'HeartRateRecord', 'sdnn': 'HeartRateRecord', 'step': 'StepsRecord', 'calorie': 'TotalCaloriesBurnedRecord'}
    # last_data_idx = [list(hour_df.to_dict('records')[-1].values())[1:][x] == None for x in range(len(hour_df.to_dict('records')[-1].values())[1:])].index(False)
    # last_data_name = list(hour_df.to_dict('records')[-1].keys())[1:][last_data_idx]
    # record_name = dict_compare[last_data_name]
    
    # 데이터 비교군 쿼리 데이터의 마지막 startTime
    # record_query = pd.to_datetime(query_one_data(user_email, record_name)['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19])
    
    # dynamodb_last_time = query_one_data(user_email)['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19].floor('h')
    
    # MongoDB 해당 유저 마지막 data ds와 dynamodb 최신 데이터 startTime + 72시간 비교 (prophet을 마지막 bpm으로부터 72개를 구했으므로)
    if (pd.to_datetime(mongo_last_ds) == pd.to_datetime(query_one_data(user_email)['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19]).floor('h') + timedelta(hours=72)) == True:
        return {'message': '동기화할 데이터가 없습니다.'}
    
    else:
        bpm_data = query_bpm_data(user_email)
        step_data = query_step_data(user_email)
        calorie_data = query_calorie_data(user_email)
        
           
        bpm_hour, bpm_day, last_ds = create_bpm_dataframe_(bpm_data)
        step_hour, step_day = create_step_dataframe_(step_data)
        calorie_hour, calorie_day = create_calorie_dataframe_(calorie_data)
        
        hour_df = pd.concat([bpm_hour, step_hour, calorie_hour], axis=0).sort_values('ds').reset_index(drop=True).groupby('ds', as_index=False).first()
        day_df = pd.concat([bpm_day, step_day, calorie_day], axis=0).sort_values('ds').reset_index(drop=True).groupby('ds', as_index=False).first()
        
        hour_df = hour_df.replace({np.nan: None})
        day_df = day_df.replace({np.nan: None})
        
        hourly_collection.insert_one({
            'user_email': user_email,
            'date': str(input_date.year) + '-' + str(input_date.month).zfill(2) + '-' + str(input_date.day).zfill(2) + ' ' + str(input_date.hour).zfill(2) + ':' + str(input_date.minute).zfill(2) + ':' + str(input_date.second).zfill(2),
            'data': hour_df.to_dict('records'),
        })
        
        daily_collection.insert_one({
            'user_email': user_email,
            'date': str(input_date.year) + '-' + str(input_date.month).zfill(2) + '-' + str(input_date.day).zfill(2) + ' ' + str(input_date.hour).zfill(2) + ':' + str(input_date.minute).zfill(2) + ':' + str(input_date.second).zfill(2),
            'data': day_df.to_dict('records'),
        })
        
        return {'message': '데이터 저장 완료'}
    
@app.get("/prediction_dates/{user_email}")
async def get_prediction_dates(user_email: str):
    dates = prediction_collection.distinct("prediction_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}

    # 시간 체크
@app.post("/check_db1")
async def check_db(request: UserEmailRequest):
    user_email = request.user_email
    
    input_date = datetime.now() + timedelta(hours=9)
    
    # analysis, predict collection 동시에 처리하기에 하나로만 처리? 
    if prediction_collection.find_one({"user_email": user_email}) == None:

        ##################### CALORIE ###################################
        # 칼로리 데이터 query 시간 체크
        start_time = time.time()
        mongo_new_calorie_data = query_latest_calorie_data(user_email)
        end_time = time.time()
        print(f'칼로리 데이터 query 걸린 시간(query_latest_calorie_data) : {(end_time - start_time):.4f}')
        
        # 칼로리 데이터프레임 만드는 시간 체크
        start_time = time.time()
        mongo_new_calorie_df = create_calorie_dataframe(mongo_new_calorie_data)
        end_time = time.time()
        print(f'칼로리 데이터프레임 만드는데 걸린 시간(create_calorie_dataframe) : {(end_time - start_time):.4f}')
        ##################### CALORIE ###################################
        
        ##################### HRV ###################################
        # 1분 BPM query 걸린 시간 체크
        start_time = time.time()
        mongo_new_data = query_latest_heart_rate_data(user_email)
        end_time = time.time()
        print(f'1분 BPM 데이터 query 걸린 시간(query_latest_heart_rate_data) : {(end_time - start_time):.4f}')
        
        # 1분 BPM 데이터프레임 만드는데 걸린 시간 체크
        start_time = time.time()
        mongo_new_df = create_heart_rate_dataframe(mongo_new_data)
        end_time = time.time()
        print(f'1분 BPM 데이터프레임 만드는 데 걸린 시간(create_heart_rate_dataframe) : {(end_time - start_time):.4f}')
        
        # 1분 BPM 데이터 feature 계산 걸린 시간 체크
        start_time = time.time()
        mongo_new_hrv_analysis = preprocess_analysis(mongo_new_df) # 분석
        end_time = time.time()
        print(f'1분 BPM 데이터 time_feature 계산 걸린 시간(preprocess_analysis) : {(end_time - start_time):.4f}')
        
        # 1분 BPM 데이터 prophet 예측 걸린 시간 체크
        start_time = time.time()
        mongo_new_forecast = predict_heart_rate(mongo_new_df) # 예측
        end_time = time.time()
        print(f'1분 BPM 데이터 prophet 예측 걸린 시간(predict_heart_rate) : {(end_time - start_time):.4f}')        
        
        ##################### HRV ###################################
        
        ##################### STEP ###################################
        # 걸음수 데이터 query 걸린 시간 체크
        start_time = time.time()
        mongo_new_step_data = query_latest_step_data(user_email)
        end_time = time.time()
        print(f'걸음수 데이터 query 걸린 시간(query_latest_step_data) : {(end_time - start_time):.4f}')
        
        # 걸음수 데이터프레임 만드는데 걸린 시간 체크
        start_time = time.time()
        mongo_new_step_df = create_step_dataframe(mongo_new_step_data)       
        end_time = time.time()
        print(f'걸음수 데이터 데이터프레임 만드는데 걸린 시간(create_step_dataframe) : {(end_time - start_time):.4f}')       
        ##################### STEP ###################################
            
        ##################### SLEEP ###################################
        # 수면 데이터 query 걸린 시간 체크
        start_time = time.time()
        mongo_new_sleep_data = query_latest_sleep_data(user_email)
        end_time = time.time()
        print(f'수면 데이터 query 걸린 시간(query_latest_sleep_data) : {(end_time - start_time):.4f}')
        
        # 수면 데이터 query 걸린 시간 체크
        start_time = time.time()
        mongo_new_sleep_df = create_sleep_dataframe(mongo_new_sleep_data)
        end_time = time.time()
        print(f'수면 데이터프레임 만드는데 걸린 시간(create_sleep_dataframe) : {(end_time - start_time):.4f}')
        ##################### SLEEP ###################################
        
        
        # 칼로리 데이터 몽고DB 저장 걸린 시간 체크
        start_time = time.time()
        save_calorie_to_mongodb(user_email, mongo_new_calorie_df, input_date) # 칼로리 데이터 저장
        end_time = time.time()
        print(f'칼로리 데이터 몽고DB 저장 걸린 시간(save_calorie_to_mongodb) : {(end_time - start_time):.4f}')
        
        # 분석 데이터 몽고DB 저장 걸린 시간 체크
        start_time = time.time()       
        save_analysis_to_mongodb(user_email, mongo_new_hrv_analysis, input_date) # 분석 데이터 저장
        end_time = time.time()
        print(f'분석 데이터 몽고DB 저장 걸린 시간(save_analysis_to_mongodb) : {(end_time - start_time):.4f}')
        
        # 예측 데이터 몽고DB 저장 걸린 시간 체크
        start_time = time.time()
        save_prediction_to_mongodb(user_email, mongo_new_forecast, input_date) # 예측 데이터 저장
        end_time = time.time()
        print(f'예측 데이터 몽고DB 저장 걸린 시간(save_prediction_to_mongodb) : {(end_time - start_time):.4f}')
        
        # 걸음수 데이터 몽고DB 저장 걸린 시간 체크
        start_time = time.time()
        save_step_to_mongodb(user_email, mongo_new_step_df, input_date) # 걸음수 데이터 저장
        end_time = time.time()
        print(f'걸음수 데이터 몽고DB 저장 걸린 시간(save_step_to_mongodb) : {(end_time - start_time):.4f}')
        
        # 수면 데이터 몽고DB 저장 걸린 시간 체크
        start_time = time.time()
        save_sleep_to_mongodb(user_email, mongo_new_sleep_df, input_date) # 수면 데이터 저장
        end_time = time.time()
        print(f'수면 데이터 몽고DB 저장 걸린 시간(save_sleep_to_mongodb) : {(end_time - start_time):.4f}')
        
        return {'message': '데이터 저장 완료'}
    
    last_data = list(prediction_collection.find({"user_email": user_email}))[-1] # 최신 데이터의 마지막 data를 선택
    #print(f'last_data: {last_data}')
    datetime_last = last_data['data'][-4321]['ds'] # 그 데이터의 예측값을 제외한 마지막 값
    last_date = str(datetime_last.year) + '-' + str(datetime_last.month).zfill(2) + '-' + str(datetime_last.day).zfill(2) + ' ' + str(datetime_last.hour).zfill(2) + ':' + str(datetime_last.minute).zfill(2) + ':' + str(datetime_last.second).zfill(2)
    
    # DynamoDB에 마지막 데이터와 collection에 -4321번째의 데이터가 같다면 최대 40000개의 query를 하지 않음
    if last_date == conv_ds(query_one_heart_rate_data(user_email)['SK']['S'].split('#')[1]) :
        return {'message': '새로운 데이터가 없습니다.'}
    # 만약 다르다면 : 새로 동기화된 데이터가 있다면..
    else:
        
        ##################### CALORIE ###################################
        mongo_new_calorie_data = query_latest_calorie_data(user_email)
        mongo_new_calorie_df = create_calorie_dataframe(mongo_new_calorie_data)
        ##################### CALORIE ###################################
        
        ##################### HRV ###################################

        mongo_new_data = query_latest_heart_rate_data(user_email)

        mongo_new_df = create_heart_rate_dataframe(mongo_new_data)

        mongo_new_hrv_analysis = preprocess_analysis(mongo_new_df) # 분석

        mongo_new_forecast = predict_heart_rate(mongo_new_df) # 예측

        ##################### HRV ###################################
                
        ##################### STEP ###################################
        mongo_new_step_data = query_latest_step_data(user_email)
        mongo_new_step_df = create_step_dataframe(mongo_new_step_data)       
        ##################### STEP ###################################
        
        ##################### SLEEP ###################################
        mongo_new_sleep_data = query_latest_sleep_data(user_email)
        mongo_new_sleep_df = create_sleep_dataframe(mongo_new_sleep_data)
        ##################### SLEEP ###################################
        
        save_calorie_to_mongodb(user_email, mongo_new_calorie_df, input_date) # 칼로리 데이터 저장
        save_analysis_to_mongodb(user_email, mongo_new_hrv_analysis, input_date) # 분석 데이터 저장
        save_prediction_to_mongodb(user_email, mongo_new_forecast, input_date) # 예측 데이터 저장
        save_step_to_mongodb(user_email, mongo_new_step_df, input_date) # 걸음수 데이터 저장
        save_sleep_to_mongodb(user_email, mongo_new_sleep_df, input_date) # 수면 데이터 저장
        
        return {'message': '데이터 저장 완료'}
    
    
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)