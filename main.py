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
    print(f'In Process Heart Rate Data : {processed_data}')
    return processed_data

def save_prediction_to_mongodb(user_email: str, prediction_data):
    korea_time = datetime.now() + timedelta(hours=9)
    prediction_collection.insert_one({
        "user_email": user_email,
        "prediction_date": str(korea_time.year) + '-' + str(korea_time.month) + '-' + str(korea_time.day) + ' ' + str(korea_time.hour) + ':' + str(korea_time.minute) + ':' + str(korea_time.second),
        "data": prediction_data.to_dict('records')
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
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

@app.get("/prediction_dates/{user_email}")
async def get_prediction_dates(user_email: str):
    dates = prediction_collection.distinct("prediction_date", {"user_email": user_email})
    return {"dates": [date.isoformat() for date in dates]}

@app.get("/prediction_data/{user_email}/{prediction_date}")
async def get_prediction_data(user_email: str, prediction_date: str):
    try:
        # URL 디코딩 및 ISO 형식 파싱
        decoded_date = unquote(prediction_date)
        date = datetime.fromisoformat(decoded_date)
        
        prediction = prediction_collection.find_one({"user_email": user_email, "prediction_date": date})
        if prediction:
            return {"data": prediction["data"]}
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

# DynamoDB의 마지막 데이터(시간)과 저장된 MongoDB의 -4321번째(3일 예측 전 마지막 데이터의 시간)와 같은지 비교
# 만약 다르다면, DynamoDB에 새로운 데이터가 있으니, DynamoDB Query 실행
# 만약 같다면, DynamoDB Query를 할 필요가 없으니, MongoDB Data만 보내줌.
@app.post("/check_db")
async def check_db(request: UserEmailRequest):
    user_email = request.user_email
    
    if prediction_collection.find_one() == None:
        # 해당 사용자의 데이터가 MongoDB에 없을 경우
        mongo_new_data = query_latest_heart_rate_data(user_email)
        mongo_new_df = create_dataframe(mongo_new_data)
        
        mongo_new_forecast = predict_heart_rate(mongo_new_df)
        save_prediction_to_mongodb(user_email, mongo_new_forecast)
        return {'message': '데이터 저장 완료'}
    
    datetime_last = prediction_collection.find()[len(prediction_collection.find_one())-1]['data'][-4321]['ds']
    last_date = datetime_last.year + '-' + datetime_last.month + '-' + datetime_last.day + ' ' + datetime_last.hour + ':' + datetime_last.minute + ':' + datetime_last.second
    
    if last_date == conv_ds(query_one_heart_rate_data(user_email)['SK']['S'].split('#')[1]) :
        # 새로 동기화된 데이터가 DynamoDB에 없을 경우
        return {'message': '새로운 데이터가 없습니다.'}
    else:
        # 새로 동기화된 데이터가 DynamoDB에 있을 경우..
        is_sync_new_data = query_latest_heart_rate_data(user_email)
        sync_new_df = create_dataframe(is_sync_new_data)
        
        snyc_new_forecast = predict_heart_rate(sync_new_df)
        save_prediction_to_mongodb(user_email, snyc_new_forecast)
        return {"message": "데이터 저장 완료"}
        

# @app.post("/analyze_and_predict")
# async def analyze_and_predict(request: UserEmailRequest):
#     user_email = request.user_email
#     items = query_latest_heart_rate_data(user_email)
#     if not items:
#         raise HTTPException(status_code=404, detail="유저 정보 없음")
#     processed_data = process_heart_rate_data(items)
#     if not processed_data:
#         raise HTTPException(status_code=404, detail="유저의 데이터가 없음")
    
#     df = pd.DataFrame(processed_data)
#     df['ds'] = pd.to_datetime(df['ds'])
    
#     forecast = predict_heart_rate(df)
#     save_prediction_to_mongodb(user_email, forecast)
#     return {"message": "분석 끝 예측 데이터 저장", "데이터 길이": len(df)}

  
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)