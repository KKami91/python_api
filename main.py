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
from pymongo.errors import BulkWriteError
from urllib.parse import unquote
import numpy as np
import math
import neurokit2 as nk
import time
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio



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
#client = MongoClient(MONGODB_URI)
client = AsyncIOMotorClient(MONGODB_URI)
db = client.get_database("heart_rate_db")


################ 데이터 각각 나눈 버전 ###############
bpm_div = db.bpm_div
step_div = db.step_div
calorie_div = db.calorie_div
sleep_div = db.sleep_div



#############################################################
#############################################################

################ 데이터 저장 테스트 ##################
bpm_test2 = db.bpm_test2
step_test2 = db.step_test2
calorie_test2 = db.calorie_test2
sleep_test2 = db.sleep_test2

rmssd = db.rmssd
sdnn = db.sdnn


bpm_test3 = db.bpm_test3
step_test3 = db.step_test3
calorie_test3 = db.calorie_test3
sleep_test3 = db.sleep_test3
rmssd3 = db.rmssd3
sdnn3 = db.sdnn3

########## dynamodb process time check ##########
@app.post("/check_db3_dynamodb")
async def check_db_query_div_dynamodb(request: UserEmailRequest):
    
    # 전체 check DB 걸린 시간 -> all_end_time - all_start_time
    all_start_time = datetime.now()
    
    user_email = request.user_email
    record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
    collection_names_div = ['bpm_test3', 'step_test3', 'calorie_test3', 'sleep_test3']

    # MongoDB 컬렉션에 데이터가 존재하는지 걸린 시간 -> exist_items_end_time - exist_items_start_time
    exist_items_start_time = datetime.now()
    exist_times = await exist_collection_div(user_email, collection_names_div)
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
    print(f'In Python ---> dynamoDB 데이터 데이터프레임 만드는데 걸린 시간 @@ : {create_df_end_time - create_df_start_time}')
    
    mongo_save_start_time = datetime.now()
    #await asyncio.gather(*[update_db_div(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))])
    await asyncio.gather(*[update_db(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))])
    mongo_save_end_time = datetime.now()
    print(f'In Python ---> MongoDB 저장 : {mongo_save_end_time - mongo_save_start_time} (2)')
    
    all_end_time = datetime.now()
    print(f'In Python ---> 전체 끝나는데 까지 걸린 시간 @@ : {all_end_time - all_start_time}')

def save_hrv(user_email, hrv_df):
    print(f'{user_email} ----> save_hrv 저장 구간 진입')
    save_hrv_start_time = datetime.now()
    rmssd_df = hrv_df[['ds', 'day_rmssd']]
    sdnn_df = hrv_df[['ds', 'day_sdnn']]

    rmssd_docs = prepare_docs_hrv(user_email, rmssd_df)
    sdnn_docs = prepare_docs_hrv(user_email, sdnn_df)

    rmssd_bulk_operations = [
        UpdateOne(
            {
                'user_email': rmssd_doc['user_email'],
                'timestamp': rmssd_doc['timestamp'],
            },
            {'$set':rmssd_doc},
            upsert=True
        ) for rmssd_doc in rmssd_docs
    ]

    sdnn_bulk_operations = [
        UpdateOne(
            {
                'user_email': sdnn_doc['user_email'],
                'timestamp': sdnn_doc['timestamp'],
            },
            {'$set':sdnn_doc},
            upsert=True
        ) for sdnn_doc in sdnn_docs
    ]

    batch_size = 10000
    rmssd_total_operations = len(rmssd_bulk_operations)
    sdnn_total_operations = len(sdnn_bulk_operations)
    
    for i in range(0, rmssd_total_operations, batch_size):
        batch = rmssd_bulk_operations[i:i+batch_size]
        rmssd3.bulk_write(batch, ordered=False)

    for i in range(0, sdnn_total_operations, batch_size):
        batch = sdnn_bulk_operations[i:i+batch_size]
        sdnn3.bulk_write(batch, ordered=False)
        
    save_hrv_end_time = datetime.now()
    print(f'{user_email} --- save_hrv 걸린 시간 : {save_hrv_end_time - save_hrv_start_time} ---> 총 길이, {len(rmssd_df)}')



def prepare_docs_hrv(user_email, df):
    return [
        {
            'user_email': user_email,
            'type': df.columns[1],
            'value': np.round(float(row[df.columns[1]]), 3),
            'timestamp': row['ds']
        }
        for row in df.to_dict('records')
    ]
    
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



# @app.post("/check_db3_div")
# async def check_db_query_div(request: UserEmailRequest):
#     check_db3_div_start_time = datetime.now()
#     user_email = request.user_email
#     record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
#     collection_names_div = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
    
#     dynamo_start_time = datetime.now()
#     exist_times = exist_collection_div(user_email, collection_names_div)
#     json_data = [new_query_div(user_email, record_names[x], exist_times[x]) for x in range(len(exist_times))]
#     df_data = [create_df_div(json_data[x]) for x in range(len(json_data))]
#     dynamo_end_time = datetime.now()
#     print(f'DynamoDB Data 불러오기 및 DataFrame 생성 (1) : {dynamo_end_time - dynamo_start_time}s')
    
#     mongo_start_time = datetime.now()
#     [update_db_div(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))]
#     mongo_end_time = datetime.now()
#     print(f'MongoDB 저장 : {mongo_end_time - mongo_start_time} (2)')
#     print(f'In Python ---> check_db3_div 끝나는데 까지 시간 @@ (1+2) : {mongo_end_time - check_db3_div_start_time}')
    
 
async def exist_collection_div(user_email, collections):
    res_idx = []
    collection_list_name = await db.list_collection_names()
    for idx in collections:
        if idx not in collection_list_name:
            await db.create_collection(idx)
            if idx == 'sleep_test3':
                await db[idx].create_index([('user_email', ASCENDING)])
                await db[idx].create_index([('timestamp_start', ASCENDING)])
                #await db[idx].create_index([('user_email', ASCENDING), ('timestamp_start', ASCENDING)])
            else:
                await db[idx].create_index([('user_email', ASCENDING)])
                await db[idx].create_index([('timestamp', ASCENDING)])
                #await db[idx].create_index([('user_email', ASCENDING), ('timestamp', ASCENDING)])
            res_idx.append('0000-00-00T00:00:00')
        else:
            collection = db[idx]
            if idx == 'sleep_test3':
                doc = await collection.find_one({'user_email': user_email}, sort=[('timestamp_start', DESCENDING)])
            else:
                doc = await collection.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
            if doc is None:
                res_idx.append('0000-00-00T00:00:00')
            else:
                if idx == 'sleep_test3':
                    print('-------', idx, '------')
                    res_idx.append(str(doc['timestamp_end']))
                else:
                    res_idx.append(str(doc['timestamp']))
    return res_idx

def new_query_div(user_email, record_name, start_time):
    # 각 컬렉션 별 query 걸린 시간 체크...
    dynamodb_query_start_time = datetime.now()
    new_items = []
    start_time = start_time
    last_evaluated_key = None
    
    print(f'in new_query_div start_time : {start_time}')
    print(f'in new_query_div record_name : {record_name}')
    
    if record_name == 'Steps':
        start_time = start_time.replace(' ', 'T')
    else:
        if str(start_time)[:4] == '0000':
            start_time = start_time
        else:
            if record_name == 'HeartRate' or record_name == 'TotalCaloriesBurned':
                print(f'여기로 들어와야 함.')
                start_time = str(pd.to_datetime(start_time) - timedelta(hours=9) + timedelta(minutes=1)).replace(' ', 'T')
                print(f'내부 start_time : {start_time}')
            else:
                print(f'여긴 안돼')
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
        
async def update_db(user_email, df, collection):
    if len(df) == 0:
        return 0
    
    print(f'user_email : {user_email}, ,df[:3] : {df[:3]}, ,collection : {collection}')
    start_doc = time.time()

    documents = prepare_docs(user_email, df, collection)
    
    end_doc = time.time()
    # print(documents)
    print(f'{user_email} - {collection} prepare_docs 걸린 시간 : {end_doc - start_doc}')
    
    is_sleep_collection = collection.startswith('sleep_')
    
    bulk_operations = [
        UpdateOne(
            {
                'user_email': doc['user_email'],
                'timestamp_start' if is_sleep_collection else 'timestamp': doc['timestamp_start' if is_sleep_collection else 'timestamp'],
            },
            {'$set': doc},
            upsert=True
        ) for doc in documents
    ]
    
    batch_size = 1000  # Reduced batch size for better performance
    total_operations = len(bulk_operations)
    total_updated = 0
    
    start_time = time.time()

    async def process_batch(batch):
        try:
            result = await eval(collection).bulk_write(batch, ordered=False)
            return result.upserted_count + result.modified_count
        except BulkWriteError as bwe:
            print(f"Bulk write error: {bwe.details}")
            return bwe.details['nUpserted'] + bwe.details['nModified']

    tasks = [
        process_batch(bulk_operations[i:i+batch_size])
        for i in range(0, total_operations, batch_size)
    ]

    results = await asyncio.gather(*tasks)
    total_updated = sum(results)
    
    end_time = time.time()
    print(f'{user_email} 사용자의 {collection} Data Bulk write 걸린 시간 --> {end_time - start_time}')
    print(f'{user_email} 사용자의 {collection} Update Db 전체 걸린 시간 --> {end_time - start_doc}')
    
    #return total_updated

async def update_db_div(user_email, df, collection):
    if len(df) == 0:
        return 0
    # prepare_docs() 구조 bpm, step, calorie, step 들어가야
    print(f'user_email : {user_email}, ,df[:3] : {df[:3]}, ,collection : {collection}')
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
        
    batch_size = 10000
    
    total_operations = len(bulk_operations)
    
    start_time = time.time()
    
    for i in range(0, total_operations, batch_size):
        batch = bulk_operations[i:i+batch_size]
        await eval(collection).bulk_write(batch, ordered=False)
        
    end_time = time.time()
    print(f'{user_email} 사용자의 {collection} Data 저장 걸린 시간 --> {end_time - start_doc}')


    
@app.get("/get_save_dates/{user_email}")
async def get_save_dates(user_email: str):
    collections = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
    return {"save_dates": [max(exist_collection_div(user_email, collections))]}

@app.get("/get_save_dates_div/{user_email}")
async def get_save_dates_div(user_email: str):
    collections = ['bpm_test3', 'step_test3', 'calorie_test3', 'sleep_test3']
    return {"save_dates": [max(await exist_collection_div(user_email, collections))]}

async def get_bpm_hour_data(user_email, start_date, end_date):
    cursor = bpm_test3.find({'user_email': user_email, 'timestamp': {'$gte': datetime.fromtimestamp(int(str(start_date)[:-3])), '$lte': datetime.fromtimestamp(int(str(end_date)[:-3]))}})
    results = []
    async for document in cursor:
        results.append(document)
    return results

@app.get("/feature_hour_div/{user_email}/{start_date}/{end_date}")
async def bpm_hour_feature(user_email: str, start_date: str, end_date: str):
    query = await get_bpm_hour_data(user_email, start_date, end_date)
    mongo_bpm_df = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in query],
        'bpm': [doc['value'] for doc in query]
    })
    print(f'----------In featureHour len bpm_df---------- {len(mongo_bpm_df)}')
    if len(mongo_bpm_df) == 0:
        return 0
    mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
    mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

    hour_group = mongo_bpm_df.groupby(mongo_bpm_df['ds'].dt.floor('h'))
    hour_hrv = hour_group.apply(calc_hrv).reset_index()
    
    hour_hrv.rename(columns={'rmssd' : 'hour_rmssd', 'sdnn' : 'hour_sdnn'}, inplace=True)
    
    return {'hour_hrv': hour_hrv[['ds', 'hour_rmssd', 'hour_sdnn']].to_dict('records')}

async def get_bpm_all_data(user_email, start_date):
    cursor = bpm_test3.find({'user_email': user_email, 'timestamp': {'$gte': start_date}})
    results = await cursor.to_list(length=None)
    return results

async def get_hrv_all_data(user_email):
    cursor_rmssd = rmssd3.find({'user_email': user_email})
    cursor_sdnn = sdnn3.find({'user_email': user_email})
    results_rmssd = await cursor_rmssd.to_list(length=None)
    results_sdnn = await cursor_sdnn.to_list(length=None)

    df_rmssd = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in results_rmssd],
        'day_rmssd': [doc['value'] for doc in results_rmssd]
    })
    df_sdnn = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in results_sdnn],
        'day_sdnn': [doc['value'] for doc in results_sdnn]
    })

    df_merged = pd.merge(df_rmssd, df_sdnn, on='ds', how='outer')

    df_merged = df_merged.sort_values('ds').reset_index(drop=True)
    
    return df_merged

# Heatmap 수정본
@app.get("/feature_day_div/{user_email}")
async def bpm_day_feature(user_email: str):
    # 추가적으로 BPM 데이터 업데이트 하는 부분도 필요.
    update_bpm_ds = await exist_collection_div(user_email, ['bpm_test3'])
    # print(update_bpm_ds)
    update_bpm_data = new_query_div(user_email, 'HeartRate', update_bpm_ds[0])
    # print(update_bpm_data)
    if len(update_bpm_data) > 0:
        update_df = create_df_div(update_bpm_data)
        await update_db_div(user_email, update_df, 'bpm_test3')
        # update_db_div(user_email, update_df, bpm_test2)
    
    
    
    rmssd_last_date = await rmssd3.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    bpm_last_date = await bpm_test3.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    
    print(rmssd_last_date)
    
    if rmssd_last_date == None:
        rmssd_last_date = {'timestamp': datetime(1,1,1)}
    # 업데이트 된 데이터가 있다면, HRV Data 마지막 timestamp <-> BPM 데이터 마지막 timestamp 비교
    if bpm_last_date['timestamp'] - rmssd_last_date['timestamp'] > timedelta(days=1):
        update_rmssd_query = await bpm_test3.find({'user_email': user_email, 'timestamp': {'$gte': rmssd_last_date['timestamp'], '$lte': bpm_last_date['timestamp']}}).to_list(length=None)
        update_rmssd_df = pd.DataFrame({
                'ds': pd.to_datetime([update_rmssd_query[x]['timestamp'] for x in range(len(update_rmssd_query))]),
                'bpm': [int(update_rmssd_query[x]['value']) for x in range(len(update_rmssd_query))]
            })
        day_group = update_rmssd_df.groupby(update_rmssd_df['ds'].dt.floor('d'))
        day_hrv = day_group.apply(calc_hrv).reset_index()

        day_hrv.rename(columns={'rmssd' : 'day_rmssd', 'sdnn' : 'day_sdnn'}, inplace=True)
        
        save_hrv(user_email, day_hrv)

    day_hrv = await get_hrv_all_data(user_email)


    return {'day_hrv': day_hrv[['ds', 'day_rmssd', 'day_sdnn']].to_dict('records')}




@app.get("/predict_minute_div/{user_email}")
async def bpm_minute_predict(user_email: str):

    start_time = datetime.now()
    
    last_bpm_data = await bpm_test3.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    query = await get_bpm_all_data(user_email, last_bpm_data['timestamp'] - timedelta(days=15))
    mongo_bpm_df = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in query],
        'bpm': [doc['value'] for doc in query]
    })
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    min_model = Prophet(
        changepoint_prior_scale=0.0001,
        seasonality_mode='multiplicative',
    )
    # min_model.add_seasonality(name='hourly', period=24, fourier_order=5)
    # min_model.add_country_holidays(country_name='KOR')
    min_model.fit(mongo_bpm_df)
    
    min_future = min_model.make_future_dataframe(periods=60*24*1, freq='min')
    min_forecast = min_model.predict(min_future)
    
    min_forecast.rename(columns={'yhat': 'min_pred_bpm'}, inplace=True)
    min_forecast['min_pred_bpm'] = np.round(min_forecast['min_pred_bpm'], 3)  
    min_forecast = min_forecast[len(min_forecast) - 60*24*1:]
    
    end_time = datetime.now()
    print(f'in predict_minute_div 걸린 시간 : {end_time - start_time} --- {len(mongo_bpm_df)}')

    return {'min_pred_bpm': min_forecast[['ds', 'min_pred_bpm']].to_dict('records')}
       
@app.get("/predict_hour_div/{user_email}")
async def bpm_hour_predict(user_email: str):

    start_time = datetime.now()
    last_bpm_data = await bpm_test3.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    query = await get_bpm_all_data(user_email, last_bpm_data['timestamp'] - timedelta(days=30))
    mongo_bpm_df = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in query],
        'bpm': [doc['value'] for doc in query]
    })
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    hour_df = mongo_bpm_df.groupby(mongo_bpm_df['ds'].dt.floor('h')).agg({'y':'mean'}).reset_index()
    hour_model = Prophet(
        changepoint_prior_scale=0.01,
        seasonality_mode='multiplicative',
    )
    hour_model.add_seasonality(name='hourly', period=24, fourier_order=5)
    hour_model.add_country_holidays(country_name='KOR')
    hour_model.fit(hour_df)

    hour_future = hour_model.make_future_dataframe(periods=72, freq='h')
    hour_forecast = hour_model.predict(hour_future)
    
    hour_forecast.rename(columns={'yhat': 'hour_pred_bpm'}, inplace=True)
    hour_forecast['hour_pred_bpm'] = np.round(hour_forecast['hour_pred_bpm'], 3)
    
    end_time = datetime.now()
    print(f'in predict_hour_div 걸린 시간 : {end_time - start_time} ---- {len(hour_df)}')

    return {'hour_pred_bpm': hour_forecast[['ds', 'hour_pred_bpm']][len(hour_forecast) - 72:].to_dict('records')}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
