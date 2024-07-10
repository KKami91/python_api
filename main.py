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
from pymongo import MongoClient
from urllib.parse import unquote
import numpy as np
import math
import neurokit2 as nk

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


# 시간을 한국 시간으로 변형
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

# 마지막 데이터로부터 40000개의 데이터만 query
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
    
# 마지막 데이터 1개만 query (DynamoDB 데이터가 새로 동기화가 되었는지 확인 -> MongoDB에 저장된 데이터와 비교를 위해)    
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

def save_analysis_to_mongodb(user_email: str, analysis_data):
    korea_time = datetime.now() + timedelta(hours=9)
    
    analysis_collection.insert_one({
        "user_email": user_email,
        "analysis_date": str(korea_time.year) + '-' + str(korea_time.month).zfill(2) + '-' + str(korea_time.day).zfill(2) + ' ' + str(korea_time.hour).zfill(2) + ':' + str(korea_time.minute).zfill(2) + ':' + str(korea_time.second).zfill(2),
        "data": analysis_data
    })

def save_prediction_to_mongodb(user_email: str, prediction_data):
    korea_time = datetime.now() + timedelta(hours=9)
    
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


@app.get("/analysis_dates/{user_email}")
async def get_analysis_dates(user_email: str):
    dates = analysis_collection.distinct("analysis_dates", {"user_email": user_email})
    return {"dates": [date for date in dates]}

@app.get("/analysis_data/{user_email}/{analysis_date}")
async def get_analysis_data(user_email: str, analysis_date: str):
    try:
        analysis = prediction_collection.find_one({"user_email": user_email, "analysis_date": analysis_date})
        if analysis:
            print('in True')         
            return {"data": analysis['data']}
        else:
            raise HTTPException(status_code=404, detail="Analysis data not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")


@app.get("/prediction_dates/{user_email}")
async def get_prediction_dates(user_email: str):
    dates = prediction_collection.distinct("prediction_date", {"user_email": user_email})
    return {"dates": [date for date in dates]}

@app.get("/prediction_data/{user_email}/{prediction_date}")
async def get_prediction_data(user_email: str, prediction_date: str):
    try:
        prediction = prediction_collection.find_one({"user_email": user_email, "prediction_date": prediction_date})
        if prediction:
            print('in True')
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

def create_dataframe(json_data):
    if not json_data:
        raise HTTPException(status_code=404, detail="유저 정보 없음")
    processed_load_data = process_heart_rate_data(json_data)
    if not processed_load_data:
        raise HTTPException(status_code=404, detail="유저의 데이터가 없음")
    
    df = pd.DataFrame(processed_load_data)
    df['ds'] = pd.to_datetime(df['ds'])
    
    return df


def preprocess_analysis(df):
    # 1시간 단위로 데이터 변경
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['hour'] = df['hour'].dt.hour
    df['minute'] = df['minute'].dt.minute
    
    df['ds_rounded'] = df['ds'].dt.floor('h')
    
    def divide_by_60000(y_list):
        return [np.round((60000 / y),1) for y in y_list]
    
    dict_temp = df.groupby('ds_rounded')['y'].apply(lambda x: divide_by_60000(x.tolist())).to_dict()

    # 키를 문자열 형식으로 변환
    dict_temp = {str(k): v for k, v in dict_temp.items()}
    
    key_list = []
    value_list = []
    
    for key in dict_temp.keys():
        key_list.append(key)
        
    for value in dict_temp.values():
        value_list.append(value)
        
    peaks_list = []
    for i in range(len(value_list)):
        peaks_list.append(nk.intervals_to_peaks(value_list[i]))
        
    res_dict = {}
    sdnn = []
    rmssd = []
    lf = []
    hf = []
    
    for i in range(len(peaks_list)):
        temp_nk = nk.hrv(peaks_list[i])
        sdnn.append(temp_nk['HRV_SDNN'])
        rmssd.append(temp_nk['HRV_RMSSD'])
        lf.append(temp_nk['HRV_LF'])
        hf.append(temp_nk['HRV_HF'])
    
    res_dict['ds'] = key_list
    res_dict['sdnn'] = sdnn
    res_dict['rmssd'] = rmssd
    res_dict['lf'] = lf
    res_dict['hf'] = hf
    
    nk_df = pd.DataFrame(res_dict)
    nk_df['ds'] = pd.to_datetime(nk_df['ds'])
    
    return nk_df

    
    

@app.post("check_db_analysis")
async def check_db_analysis(request: UserEmailRequest):
    user_email = request.user_email
    
    if analysis_collection.find_one({"user_email": user_email}) == None:
        print('In Analysis Check DB : ', user_email)
        mongo_new_data_analysis = query_latest_heart_rate_data(user_email)
        print('In Analysis Check DB After query_latest_hrv_data: ', mongo_new_data_analysis)
        mongo_new_df_analysis = create_dataframe(mongo_new_data_analysis)
        print('In Analysis Check DB After create_dataframe: ', mongo_new_df_analysis)
        mongo_new_preprocess_analysis = preprocess_analysis(mongo_new_df_analysis)
        print('In Analysis Check DB After preprocess_analysis: ', mongo_new_preprocess_analysis)
        mongo_new_nk_analysis = preprocess_analysis(mongo_new_preprocess_analysis)
        print('In Analysis Check DB After preprocess_analysis: ', mongo_new_nk_analysis)
                
        save_prediction_to_mongodb(user_email, mongo_new_nk_analysis)
        return {'message': '데이터 저장 완료'}  
        
    last_data = list(prediction_collection.find({"user_email": user_email}))[-1]
    datetime_last = last_data['data'][-4321]['ds']
    last_date = str(datetime_last.year) + '-' + str(datetime_last.month).zfill(2) + '-' + str(datetime_last.day).zfill(2) + ' ' + str(datetime_last.hour).zfill(2) + ':' + str(datetime_last.minute).zfill(2) + ':' + str(datetime_last.second).zfill(2)
    
    if last_date == conv_ds(query_one_heart_rate_data(user_email)['SK']['S'].split('#')[1]) :
        # 새로 동기화된 데이터가 DynamoDB에 없을 경우
        return {'message': '새로운 데이터가 없습니다.'}
    else:
        # 새로 동기화된 데이터가 DynamoDB에 있을 경우..
        mongo_new_data_analysis = query_latest_heart_rate_data(user_email)
        mongo_new_df_analysis = create_dataframe(mongo_new_data_analysis)
        
        mongo_new_preprocess_analysis = preprocess_analysis(mongo_new_df_analysis)
        
        mongo_new_nk_analysis = preprocess_analysis(mongo_new_preprocess_analysis)
                
        save_prediction_to_mongodb(user_email, mongo_new_nk_analysis)
        return {'message': '데이터 저장 완료'}       

# DynamoDB의 마지막 데이터(시간)과 저장된 MongoDB의 -4321번째(3일 예측 전 마지막 데이터의 시간)와 같은지 비교
# 만약 다르다면, DynamoDB에 새로운 데이터가 있으니, DynamoDB Query 실행
# 만약 같다면, DynamoDB Query를 할 필요가 없으니, MongoDB Data만 보내줌.
@app.post("/check_db_predict")
async def check_db_predict(request: UserEmailRequest):
    user_email = request.user_email
    
    if prediction_collection.find_one({"user_email": user_email}) == None:
        # 해당 사용자의 데이터가 MongoDB에 없을 경우
        mongo_new_data_predict = query_latest_heart_rate_data(user_email)
        mongo_new_df_predict = create_dataframe(mongo_new_data_predict)
        
        mongo_new_forecast = predict_heart_rate(mongo_new_df_predict)
        
        save_prediction_to_mongodb(user_email, mongo_new_forecast)
        return {'message': '데이터 저장 완료'}
    
    last_data = list(prediction_collection.find({"user_email": user_email}))[-1]
    datetime_last = last_data['data'][-4321]['ds']
    last_date = str(datetime_last.year) + '-' + str(datetime_last.month).zfill(2) + '-' + str(datetime_last.day).zfill(2) + ' ' + str(datetime_last.hour).zfill(2) + ':' + str(datetime_last.minute).zfill(2) + ':' + str(datetime_last.second).zfill(2)
    
    if last_date == conv_ds(query_one_heart_rate_data(user_email)['SK']['S'].split('#')[1]) :
        # 새로 동기화된 데이터가 DynamoDB에 없을 경우
        return {'message': '새로운 데이터가 없습니다.'}
    else:
        # 새로 동기화된 데이터가 DynamoDB에 있을 경우..
        mongo_new_data_predict = query_latest_heart_rate_data(user_email)
        mongo_new_df_predict = create_dataframe(mongo_new_data_predict)
        
        mongo_new_forecast = predict_heart_rate(mongo_new_df_predict)
        
        save_prediction_to_mongodb(user_email, mongo_new_forecast)
        return {'message': '데이터 저장 완료'}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)