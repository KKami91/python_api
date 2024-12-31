from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from pydantic import BaseModel
import boto3
import pandas as pd
from prophet import Prophet
import datetime
from botocore.exceptions import ClientError
from datetime import datetime, timedelta, timezone
from dateutil import relativedelta
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
from dateutil.parser import parse
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import gc




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

rmssd_test = db.rmssd_test
sdnn_test = db.sdnn_test

bpm_test3 = db.bpm_test3
step_test3 = db.step_test3
calorie_test3 = db.calorie_test3
sleep_test3 = db.sleep_test3

# bpm = db.bpm
# step = db.step
# calorie = db.calorie
# sleep = db.sleep

bpm5 = db.bpm5
step5 = db.step5
calorie5 = db.calorie5
sleep5 = db.sleep5
rmssd5 = db.rmssd5
sdnn5 = db.sdnn5

rmssdt = db.rmssdt
sdnnt = db.sdnnt

####################### 저장 방식 변경 ###################
bpm = db.bpm
step = db.step
calorie = db.calorie
sleep = db.sleep

hour_rmssd = db.hour_rmssd
hour_sdnn = db.hour_sdnn

day_rmssd = db.day_rmssd
day_sdnn = db.day_sdnn
user_info = db.user_info
#########################################################
def configure_matplotlib():
    import matplotlib.font_manager as fm
    
    # 사용 가능한 폰트 목록 확인
    font_list = [f.name for f in fm.fontManager.ttflist]
    print("Available fonts:", font_list)  # 로그 확인용
    
    # 폰트 설정 시도 (우선순위대로)
    font_candidates = ['NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic']
    selected_font = None
    
    for font in font_candidates:
        if font in font_list:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
    else:
        # 폰트를 찾지 못한 경우 직접 폰트 파일 설정
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 적절한 경로로 수정
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['savefig.dpi'] = 300

async def user_info_update():
    results = [dynamodb.scan(IndexName='email-timestamp-index', TableName=TABLE_NAME)][0]['Items']
    
    for x in range(len(results)):
        user_data = {
            'user_email': results[x]['email']['S'], 
            'user_name': results[x]['name']['S'], 
            'user_gender': results[x]['gender']['S'],
            'user_height': results[x]['height']['N'],
            'user_weight': results[x]['weight']['N'],
            'user_smoke': results[x]['smokingStatus']['S'],
            'user_birth': results[x]['dob']['S']
        }
        
        user_info.update_one(
            {'user_email': user_data['user_email']},
            {'$set': user_data},
            upsert=True
        )
        
######## 걸음수 분석 #########
def step_weekday_average(df):
    result = {}
    print('in step def :', df)
    for weekday, weekday_group in df.groupby('weekdays'):
        Q1 = np.quantile(weekday_group['step'], 0.25)
        Q3 = np.quantile(weekday_group['step'], 0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        cleaned_data = weekday_group[weekday_group['step'] <= upper_bound]
        unique_dates = cleaned_data['timestamp'].dt.normalize().nunique()
        result[weekday] = np.round(cleaned_data['step'].sum() / unique_dates, 1)
    return result

def step_hour_average(df):
    result = {}
    for hour, hour_group in df.groupby('hours'):
        Q1 = np.quantile(hour_group['step'], 0.25)
        Q3 = np.quantile(hour_group['step'], 0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        cleaned_data = hour_group[hour_group['step'] <= upper_bound]
        unique_dates = cleaned_data['timestamp'].dt.normalize().nunique()
        result[hour] = np.round(cleaned_data['step'].sum() / unique_dates, 1)
    return result

@app.post("/user_analysis_step/{user_email}")
async def plot_user_analysis_step(user_email: str):
    user_info_data = await user_info.find_one({'user_email': user_email})
    user_name = user_info_data['user_name']
    user_height = user_info_data['user_height']
    user_weight = user_info_data['user_weight']
    user_bmi = np.round(int(user_weight)/((int(user_height) / 100) * 2), 3)
    user_gender = user_info_data['user_gender'][:1]
    user_birth = datetime.strptime(user_info_data['user_birth'], "%Y-%m-%d")
    today = datetime.now()
    user_age = today.year - user_birth.year
    if (today.month, today.day) < (user_birth.month, user_birth.day):
        user_age -= 1
    user_step_data = await step.find({'user_email': user_email}).to_list(length=None)
    step_df = pd.DataFrame({
        'timestamp': [user_step_data[x]['timestamp'] for x in range(len(user_step_data))],
        'step': [user_step_data[x]['value'] for x in range(len(user_step_data))]
    })
    step_df['timestamp'] = pd.to_datetime(step_df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    step_df['hours'] = step_df['timestamp'].dt.hour
    
    week_dict = {0:'월', 1:'화', 2:'수', 3:'목', 4:'금', 5:'토', 6:'일'}
    step_df['weekdays'] = step_df['timestamp'].dt.weekday.map(week_dict)
    hourly_average = step_hour_average(step_df)
    weekday_average = step_weekday_average(step_df)
    print('-----weekday_average-----')
    print(weekday_average)
    
    
    sorted_weekday = ['월', '화', '수', '목', '금', '토', '일']
    # conv_weekday_average = {day: weekday_average[day] for day in sorted_weekday if day in weekday_average}
    sorted_weekday_averages = {day: weekday_average[day] for day in sorted_weekday if day in weekday_average}
    
    hours = hourly_average.keys()
    hours_steps = hourly_average.values()
    
    print('before weekdays, weekdays_steps')
    print(sorted_weekday_averages)
    weekdays = sorted_weekday_averages.keys()
    weekdays_steps = sorted_weekday_averages.values()
    print('after weekdays, weekdays_steps')
    print(weekdays, weekdays_steps)
    
    ##### 걸음 수 <-> BPM ######
    bpm_df = await get_bpm_df(user_email)
    max_step = step_df['step'].max()
    result_step = {}
    result_bpm = {}
    for i in range(1, int(np.ceil(max_step / 10)) + 1):
        temp_step = step_df[(step_df['step'] >= 1 + (10 * (i - 1))) & (step_df['step'] <= 10 * i)].reset_index(drop=True)
        result_step[i * 10] = temp_step
        temp_bpm = bpm_df[bpm_df['timestamp'].isin(temp_step['timestamp'])]
        result_bpm[i * 10] = temp_bpm
    keys_list = list(result_bpm.keys())
    result_bpm_df = pd.DataFrame({
        'range': [f'{x - 9} ~ {x}' for x in keys_list],
        'average_bpm': [np.round(np.mean(result_bpm[x]['bpm']), 2) for x in keys_list]
    })
    range_list = list(result_bpm_df['range'])
    average_bpm_list = list(result_bpm_df['average_bpm'])
        
    try:
        configure_matplotlib()
        # fig = plt.figure(figsize=(20, 16), constrained_layout=True)
        fig = plt.figure(figsize=(30, 24))
        # gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.1)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1.2])
        plt.rc('figure', titlesize=15)
        plt.gcf().subplots_adjust(top=0.9, left=0.2, bottom=0.1, right=0.85, hspace=0.4)
        
        #### 시간별 ####
        ax1 = fig.add_subplot(gs[0])
        bar1 = ax1.bar(hours, hours_steps)
        for rect in bar1:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size=12)
        ax1.set_title(f'{user_email}, {user_name}({user_gender}), {user_age}세, {user_height}cm, {user_weight}kg, {user_bmi}(kg/m^2 = bmi)\n\n시간별 평균 걸음 수',
                        fontsize=20, 
                        pad=20,
                        linespacing=1.5)
        ax1.set_ylabel('걸음 수', fontsize=16)
        ax1.set_xlabel('시간', fontsize=16)
        ax1.set_xticks(ticks=range(0,24))
        
        #### 요일별 ####
        ax2 = fig.add_subplot(gs[1])
        print(weekdays, weekdays_steps)
        bar2 = ax2.bar(weekdays, weekdays_steps)
        for rect in bar2:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size=12)
        ax2.set_title(f'일별 평균 걸음 수',
                      fontsize=20,
                      pad=20,
                      linespacing=1.5)
        ax2.set_ylabel('걸음 수', fontsize=16)
        ax2.set_xlabel('요일', fontsize=16)
        
        ax3 = fig.add_subplot(gs[2])
        bar3 = ax3.bar(range_list, average_bpm_list)
        for rect in bar3:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size=12)
        ax3.set_title('걸음 수 범위별 평균 심박수',
                      fontsize=20,
                      pad=20,
                      linespacing=1.5)
        ax3.set_ylabel('심박수', fontsize=16)
        ax3.set_xlabel('걸음 수 범위', fontsize=16)
        ax3.set_xticks(range(len(range_list)))  # X축 위치 설정
        ax3.set_xticklabels(range_list, rotation=45, ha='center')  # 레이블과 회전 설정
        
        
    
        # 이미지 PNG로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0.5, dpi=100)
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"error generating plot : {str(e)}")
        raise HTTPException(status_code=500, detail="error generating plot")
    finally:
        plt.close('all')
        gc.collect()

######## 수면 stage별 평균 심박수 ######
def analyze_sleep_heart_rate(user_sleep_dict, user_bpm_dict):
    sleep_hr_analysis = {}
    
    for user_email in user_sleep_dict.keys():
        sleep_df = user_sleep_dict[user_email]
        bpm_df = user_bpm_dict[user_email]
        
        # 수면 중 평균 심박수 계산
        sleep_periods = []
        for _, row in sleep_df.iterrows():
            mask = (bpm_df['timestamp'] >= row['timestamp_start']) & \
                  (bpm_df['timestamp'] <= row['timestamp_end'])
            
            # 각 수면 기간의 데이터 수집
            if bpm_df.loc[mask, 'bpm'].size > 0:  # 데이터가 있는 경우만
                sleep_periods.append({
                    'stage': row['stage'],
                    'mean_hr': bpm_df.loc[mask, 'bpm'].mean(),
                    'start_time': row['timestamp_start'],
                    'duration': (row['timestamp_end'] - row['timestamp_start']).total_seconds() / 3600
                })
        
        # 수면 단계별 평균 심박수 분석
        stage_analysis = {}
        for stage in set([p['stage'] for p in sleep_periods]):
            stage_periods = [p for p in sleep_periods if p['stage'] == stage]
            if stage_periods:
                stage_analysis[stage] = {
                    'mean_hr': np.mean([p['mean_hr'] for p in stage_periods]),
                    'std_hr': np.std([p['mean_hr'] for p in stage_periods]),
                    'total_duration': sum([p['duration'] for p in stage_periods])
                }
        
        sleep_hr_analysis[user_email] = {
            'stage_analysis': stage_analysis,
            'sleep_periods': sleep_periods
        }
    
    return sleep_hr_analysis
        
######## 수면 분석 함수 ###########
def analyze_sleep_patterns(user_sleep_dict):
    # 각 사용자별 수면 단계 분포
    sleep_stage_distribution = {}
    for user_email, sleep_df in user_sleep_dict.items():
        # 각 수면 단계별 시간 계산
        stage_durations = sleep_df.groupby('stage').apply(
            lambda x: ((x['timestamp_end'] - x['timestamp_start']).dt.total_seconds()).sum()
        ) / 3600  # 시간 단위로 변환
        sleep_stage_distribution[user_email] = stage_durations
    
    data_quality = {}
    for user_email, sleep_df in user_sleep_dict.items():
        # 실제 수면 기록이 있는 날짜 수
        recorded_days = sleep_df['timestamp_start'].dt.date.nunique()
        data_quality[user_email] = {
            'recorded_days': recorded_days,
        }
        
    return sleep_stage_distribution, data_quality

def int_to_str_time(int_time):
    hours = int(int_time)
    float_minutes = (int_time - hours) * 60
    minutes = int(float_minutes)
    seconds = (float_minutes - minutes) * 60
    return f"{hours}시간 {minutes}분 {int(np.round(seconds,0))}초"

@app.post("/user_analysis_sleep/{user_email}")
async def plot_user_analysis_sleep(user_email: str):
    print('---------1---------')
    user_info_data = await user_info.find_one({'user_email': user_email})
    print('---------2--------')
    user_name = user_info_data['user_name']
    user_height = user_info_data['user_height']
    user_weight = user_info_data['user_weight']
    user_bmi = np.round(int(user_weight)/((int(user_height) / 100) * 2), 3)
    user_gender = user_info_data['user_gender'][:1]
    user_birth = datetime.strptime(user_info_data['user_birth'], "%Y-%m-%d")
    today = datetime.now()
    user_age = today.year - user_birth.year
    if (today.month, today.day) < (user_birth.month, user_birth.day):
        user_age -= 1
    print('---------3---------')
    # 수면 전용
    user_sleep_data = await sleep.find({'user_email': user_email}).to_list(length=None)
    sleep_df = pd.DataFrame({
        'timestamp_start': [user_sleep_data[x]['timestamp_start'] for x in range(len(user_sleep_data))],
        'timestamp_end': [user_sleep_data[x]['timestamp_end'] for x in range(len(user_sleep_data))],
        'stage': [user_sleep_data[x]['value'] for x in range(len(user_sleep_data))],
    })
    sleep_df['timestamp_start'] = pd.to_datetime(sleep_df['timestamp_start']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    sleep_df['timestamp_end'] = pd.to_datetime(sleep_df['timestamp_end']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    sleep_data_analysis = analyze_sleep_patterns({user_email : sleep_df})
    
    bpm_df = await get_bpm_df(user_email)
    sleep_bpm_analysis = analyze_sleep_heart_rate({user_email: sleep_df}, {user_email: bpm_df})
    sleep_bpm_stat = sleep_bpm_analysis[user_email]['stage_analysis']
    # total_sleep_time = sum(item['total_duration'] for item in sleep_bpm_stat.values())
    print(sleep_bpm_stat, '@@@@@@@@@@@')
    print(sleep_bpm_stat.items(), '!!!!!!')
    
    
    try:
        configure_matplotlib()
        stage_dict = {1: 'Awake', 4: 'Light', 5: 'Deep', 6: 'Rem'}
        mean_hr = [item['mean_hr'] for item in sleep_bpm_stat.values()]
        stages = [stage_dict[item] for item in sleep_bpm_stat.keys()]
        total_duration = [item['total_duration'] for item in sleep_bpm_stat.values()]
        total_hours = sum(total_duration)
        total_sleep_time = sum(total_duration)
        stage_percentages = np.round(np.array(total_duration) / total_sleep_time * 100, 2)
        labels = [f'{stages[x]}\n평균 심박수 : {np.round(mean_hr[x],2)}' for x in range(len(stages))]

        plt.figure(figsize=(10,8))
        plt.pie(stage_percentages,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'ha':'center', 'va':'center'},
                labeldistance=1.3)
        plt.title(f"{user_email}, {user_name}({user_gender}), {user_age}세, {user_height}cm, {user_weight}kg, {user_bmi}(kg/m^2 = bmi) \n\n 수면 분석\n 총 수면 시간 : {int_to_str_time(total_hours)}, 전체 기록 기간 : {sleep_data_analysis[1][user_email]['recorded_days']}일",
                  fontsize=14,
                  loc='center',
                  pad=40,
                  linespacing=1.5)
        plt.gcf().subplots_adjust(top=0.75)
        
        plt.legend(stages, title="Sleep Stages", loc=(1,0.8))
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0.5, dpi=180)
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"error generating plot : {str(e)}")
        raise HTTPException(status_code=500, detail="error generating plot")
    finally:
        plt.close('all')
        gc.collect()
    
    # try:
    #     for user_email, stages in sleep_data_analysis[0].items():
    #         # 총 시간 계산
    #         total_hours = stages.sum()
    #         # 비율 계산
    #         stage_percentages = (stages / total_hours) * 100
    #         stage_dict = {1: 'Awake', 4: 'Light', 5: 'Deep', 6: 'Rem'}
            
    #         # 파이 차트 생성
            
    #         plt.pie(stage_percentages, 
    #                 labels=[f'{stage_dict[stage]}' for stage, percentage in stage_percentages.items()],
    #                 autopct='%1.1f%% \n{}',
    #                 startangle=90)
    #         #plt.title(f"수면 분석 - {user_email} \n {user_email}, {user_name}({user_gender}), {user_age}세, {user_height}cm, {user_weight}kg, {user_bmi}(kg/m^2 = bmi) \n 총 수면 시간 : {np.round(total_hours,2)}시간, 전체 기록 기간 : {sleep_data_analysis[1][user_email]['recorded_days']}일", fontsize=8)
    #         plt.title(f"수면 분석 - {user_email} \n {user_email}, {user_name}({user_gender}), {user_age}세, {user_height}cm, {user_weight}kg, {user_bmi}(kg/m^2 = bmi) \n 총 수면 시간 : {int_to_str_time(total_hours)}, 전체 기록 기간 : {sleep_data_analysis[1][user_email]['recorded_days']}일", fontsize=8)
    #             # 이미지 PNG로 변환
    #     plt.legend(loc=(1,0.6))
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='png', pad_inches=0.5)
    #     buf.seek(0)
    #     plt.close()
        
    #     return StreamingResponse(buf, media_type="image/png")
    # except Exception as e:
    #     print(f"error generating plot : {str(e)}")
    #     raise HTTPException(status_code=500, detail="error generating plot")
    # finally:
    #     plt.close('all')
    #     gc.collect()
        
        
async def get_bpm_df(user_email: str):
    user_bpm_data = await bpm.find({'user_email': user_email}).to_list(length=None)
    bpm_df = pd.DataFrame({
        'timestamp': [user_bpm_data[x]['timestamp'] for x in range(len(user_bpm_data))],
        'bpm': [user_bpm_data[x]['value'] for x in range(len(user_bpm_data))]
    })  
    bpm_df['timestamp'] = pd.to_datetime(bpm_df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)  
    return bpm_df

@app.post("/user_analysis_bpm/{user_email}")
async def plot_user_analysis_bpm(user_email: str):
    user_info_data = await user_info.find_one({'user_email': user_email})
    user_name = user_info_data['user_name']
    user_height = user_info_data['user_height']
    user_weight = user_info_data['user_weight']
    user_bmi = np.round(int(user_weight)/((int(user_height) / 100) * 2), 3)
    user_gender = user_info_data['user_gender'][:1]
    user_birth = datetime.strptime(user_info_data['user_birth'], "%Y-%m-%d")
    today = datetime.now()
    user_age = today.year - user_birth.year
    if (today.month, today.day) < (user_birth.month, user_birth.day):
        user_age -= 1
    
    # user_bpm_data = await bpm.find({'user_email': user_email}).to_list(length=None)
    # #user_hour_rmssd_data = await hour_rmssd.find({'user_email': user_email})
    # #user_hour_sdnn_data = await hour_sdnn.find({'user_email': user_email})
    
    # bpm_df = pd.DataFrame({
    #     'timestamp': [user_bpm_data[x]['timestamp'] for x in range(len(user_bpm_data))],
    #     'bpm': [user_bpm_data[x]['value'] for x in range(len(user_bpm_data))]
    # })
    
    # bpm_df['timestamp'] = pd.to_datetime(bpm_df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    
    bpm_df = await get_bpm_df(user_email)
    
    # 시간대별 평균 BPM plot 전용
    bpm_df['hour_float'] = bpm_df['timestamp'].dt.hour + bpm_df['timestamp'].dt.minute / 60.0
    bins = np.arange(0, 24.5, 0.5)
    labels = [f"{int(x)}:{int((x%1)*60):02d}" for x in bins[:-1]]
    bpm_df['time_bin'] = pd.cut(bpm_df['hour_float'], bins=bins, labels=labels, right=False)
    hourly_bpm = bpm_df.groupby('time_bin')['bpm'].agg(['mean', 'std']).reset_index()
    
    # 심박수 히트맵 전용
    bpm_df['date'] = bpm_df['timestamp'].dt.date
    bpm_df['hour'] = bpm_df['timestamp'].dt.hour
    vmin = min(bpm_df['bpm'])
    vmax = max(bpm_df['bpm'])

    
    # plot
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 20))
    try:
        # 배포 전용 함수
        configure_matplotlib()
        fig = plt.figure(figsize=(20, 25), constrained_layout=True)
        # fig = plt.figure(figsize=(16, 25))
        # gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.1)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.05, 1.2])
        plt.rc('figure', titlesize=15)
        plt.gcf().subplots_adjust(top=0.85, left=0.2, bottom=0.1, right=0.85, hspace=0.4)
        
        
        # 시간대별 평균 BPM plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(range(len(hourly_bpm)), hourly_bpm['mean'], '-o', label='평균 심박수')
        ax1.fill_between(range(len(hourly_bpm)), hourly_bpm['mean'] - hourly_bpm['std'], hourly_bpm['mean'] + hourly_bpm['std'], alpha=0.2)
        ax1.set_xticks(range(len(hourly_bpm)))
        ax1.set_xticklabels(hourly_bpm['time_bin'], rotation=45)
        ax1.set_title(f'{user_email}, {user_name}({user_gender}), {user_age}세, {user_height}cm, {user_weight}kg, {user_bmi}(kg/m^2 = bmi)\n\n시간대 평균 심박수',
                      fontsize=20, 
                      pad=20,
                      linespacing=1.5)
        ax1.set_ylabel('심박수', fontsize=14)
        ax1.set_xlabel('시간', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=20)
        
        # 날짜와 시간별 평균 BPM Heatmap
        pivot_table = bpm_df.pivot_table(values='bpm', index='date', columns='hour', aggfunc='mean')
        ax2 = fig.add_subplot(gs[2])
        # heatmap = sns.heatmap(pivot_table, cmap='YlOrRd', xticklabels=True, yticklabels=True, vmin=vmin+10, vmax=vmax-10, ax=ax2, cbar_kws={'use_gridspec': True})
        sns.heatmap(pivot_table, cmap='YlOrRd', xticklabels=True, yticklabels=True, vmin=vmin+10, vmax=vmax-10, ax=ax2, cbar_kws={'use_gridspec': True})
        plt.gcf().subplots_adjust(top=0.85, left=0.2, bottom=0.1, right=0.85, hspace=0.4)
        #cbar = heatmap.collections[0].colorbar
        #cbar.ax.set_position([0.87, 0.1, 0.03, 0.3])
        ax2.set_title(f'날짜 - 시간대 평균 심박수 히트맵', fontsize=20, pad=20)
        ax2.set_xlabel('시간', fontsize=14)
        ax2.set_ylabel('날짜', fontsize=14)
        
        
        # 이미지 PNG로 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', pad_inches=0.5, dpi=80)
        buf.seek(0)
        plt.close()
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        print(f"error generating plot : {str(e)}")
        raise HTTPException(status_code=500, detail="error generating plot")
    finally:
        plt.close('all')
        gc.collect()

@app.post("/user_data")
async def get_user_data():
    collection_names = await db.list_collection_names()
    if 'user_info' not in collection_names:
        await user_info.create_index([('user_email', ASCENDING)])
        await user_info.create_index([('user_smoke', ASCENDING)])
        await user_info_update()
        return [{'user_update': 'Yes'}]
    else:
        researcher_count = dynamodb.scan(IndexName='email-timestamp-index', TableName=TABLE_NAME, Select='COUNT')['Count']
        # mongodb smoke, non smoke count로 비교
        smoke_count = await user_info.count_documents({'user_smoke': '흡연'})
        nonsmoke_count = await user_info.count_documents({'user_smoke': '비흡연'})
        if researcher_count == smoke_count + nonsmoke_count:
            return [{'user_update': 'No'}]
        else:
            await user_info_update()
            return [{'user_update': 'Yes'}]

# @app.post("/check_db3_dynamodb")
# async def check_db_query_div_dynamodb(request: UserEmailRequest):
#     print('check_db ;;;')
#     # 전체 check DB 걸린 시간 -> all_end_time - all_start_time
#     all_start_time = datetime.now()
    
#     user_email = request.user_email
#     record_names = ['HeartRate', 'Steps', 'TotalCaloriesBurned', 'SleepSession']
#     #collection_names_div = ['bpm_test3', 'step_test3', 'calorie_test3', 'sleep_test3']
#     collection_names_div = ['bpm', 'step', 'calorie', 'sleep']
    

#     # MongoDB 컬렉션에 데이터가 존재하는지 걸린 시간 -> exist_items_end_time - exist_items_start_time
#     exist_items_start_time = datetime.now()
#     exist_times = await exist_collection_div(user_email, collection_names_div)
#     exist_items_end_time = datetime.now()
#     print(f'In Python ---> MongoDB 컬렉션 데이터 존재 유무 체크 걸린 시간 @@ : {exist_items_end_time - exist_items_start_time}')
    
#     print(exist_times)
#     # DynamoDB 데이터 query 걸린 시간 
#     query_start_time = datetime.now()
#     json_data = [new_query_div(user_email, record_names[x], exist_times[x]) for x in range(len(exist_times))]
#     query_end_time = datetime.now()
#     print(f'In Python ---> dynamoDB 데이터 query 걸린 시간 @@ : {query_end_time - query_start_time}')
#     print(f'{len(json_data[0])}, {len(json_data[1])}, {len(json_data[2])}, {len(json_data[3])}')

#     # DataFrame 만드는데 걸린 시간 
#     create_df_start_time = datetime.now()
#     df_data = [create_df_div(json_data[x]) for x in range(len(json_data))]
#     create_df_end_time = datetime.now()
#     print(f'In Python ---> dynamoDB 데이터 데이터프레임 만드는데 걸린 시간 @@ : {create_df_end_time - create_df_start_time}')
    
#     mongo_save_start_time = datetime.now()
#     await asyncio.gather(*[update_db(user_email, df_data[x], collection_names_div[x]) for x in range(len(df_data))])
#     mongo_save_end_time = datetime.now()
#     print(f'In Python ---> MongoDB 저장 : {mongo_save_end_time - mongo_save_start_time} (2)')
    
#     all_end_time = datetime.now()
#     print(f'In Python ---> 전체 끝나는데 까지 걸린 시간 @@ : {all_end_time - all_start_time}')

# def save_hrv(user_email, hrv_df):
#     print(f'{user_email} ----> save_hrv 저장 구간 진입')
#     save_hrv_start_time = datetime.now()
#     rmssd_df = hrv_df[['ds', 'day_rmssd']]
#     sdnn_df = hrv_df[['ds', 'day_sdnn']]

#     rmssd_docs = prepare_docs_hrv(user_email, rmssd_df)
#     sdnn_docs = prepare_docs_hrv(user_email, sdnn_df)

#     rmssd_bulk_operations = [
#         UpdateOne(
#             {
#                 'user_email': rmssd_doc['user_email'],
#                 'timestamp': rmssd_doc['timestamp'],
#             },
#             {'$set':rmssd_doc},
#             upsert=True
#         ) for rmssd_doc in rmssd_docs
#     ]

#     sdnn_bulk_operations = [
#         UpdateOne(
#             {
#                 'user_email': sdnn_doc['user_email'],
#                 'timestamp': sdnn_doc['timestamp'],
#             },
#             {'$set':sdnn_doc},
#             upsert=True
#         ) for sdnn_doc in sdnn_docs
#     ]

#     batch_size = 10000
#     rmssd_total_operations = len(rmssd_bulk_operations)
#     sdnn_total_operations = len(sdnn_bulk_operations)
    
#     for i in range(0, rmssd_total_operations, batch_size):
#         batch = rmssd_bulk_operations[i:i+batch_size]
#         #rmssd_test.bulk_write(batch, ordered=False)
#         rmssd.bulk_write(batch, ordered=False)

#     for i in range(0, sdnn_total_operations, batch_size):
#         batch = sdnn_bulk_operations[i:i+batch_size]
#         # sdnn_test.bulk_write(batch, ordered=False)
#         sdnn.bulk_write(batch, ordered=False)
        
#     save_hrv_end_time = datetime.now()
#     print(f'{user_email} --- save_hrv 걸린 시간 : {save_hrv_end_time - save_hrv_start_time} ---> 총 길이, {len(rmssd_df)}')



# def prepare_docs_hrv(user_email, df):
#     return [
#         {
#             'user_email': user_email,
#             'type': df.columns[1],
#             'value': np.round(float(row[df.columns[1]]), 3),
#             'timestamp': row['ds']
#         }
#         for row in df.to_dict('records')
#     ]
    
# def calc_hrv(group_):
#     rr_intervals = 60000/group_['bpm'].values
#     if len(rr_intervals) == 1:
#         return pd.Series({
#             'rmssd': 0,
#             'sdnn': 0,
#         })
#     peaks = nk.intervals_to_peaks(rr_intervals)
#     hrv = nk.hrv_time(peaks)
#     return pd.Series({
#         'rmssd': np.round(hrv['HRV_RMSSD'].values[0], 3),
#         'sdnn': np.round(hrv['HRV_SDNN'].values[0], 3),
#     })


 
# async def exist_collection_div(user_email, collections):
#     res_idx = []
#     collection_list_name = await db.list_collection_names()
#     for idx in collections:
#         if idx not in collection_list_name:
#             await db.create_collection(idx)
#             #if idx == 'sleep_test3':
#             # if idx == 'sleep':
#             if idx == 'sleep':
#                 await db[idx].create_index([('user_email', ASCENDING)])
#                 await db[idx].create_index([('timestamp_start', ASCENDING)])
#             else:
#                 await db[idx].create_index([('user_email', ASCENDING)])
#                 await db[idx].create_index([('timestamp', ASCENDING)])
#             res_idx.append('0000-00-00T00:00:00')
#         else:
#             collection = db[idx]
#             #if idx == 'sleep_test3':
#             #if idx == 'sleep':
#             if idx == 'sleep':
#                 doc = await collection.find_one({'user_email': user_email}, sort=[('timestamp_start', DESCENDING)])
#             else:
#                 doc = await collection.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
#             print(idx)
#             if doc is None:
#                 res_idx.append('0000-00-00T00:00:00')
#             else:
#                 # if idx == 'sleep':
#                 if idx == 'sleep':
#                 #if idx == 'sleep_test3':
#                     print('-------', idx, '------')
#                     res_idx.append(str(doc['timestamp_end']))
#                 else:
#                     res_idx.append(str(doc['timestamp']))
#         print('#@@#@#$@#$@#$@# ',res_idx)
#     return res_idx

# def new_query_div(user_email, record_name, start_time):
#     # 각 컬렉션 별 query 걸린 시간 체크...
#     dynamodb_query_start_time = datetime.now()
#     new_items = []
#     start_time = start_time
#     last_evaluated_key = None

#     if record_name == 'Step': 
#         start_time = start_time.replace(' ', 'T')
#     else:
#         if str(start_time)[:4] == '0000':
#             start_time = start_time
#         else:
#             if record_name == 'HeartRate' or record_name == 'TotalCaloriesBurned':
                
#                 start_time = str(pd.to_datetime(start_time) + timedelta(minutes=1)).replace(' ', 'T')
                
#             else:
                
#                 start_time = str(pd.to_datetime(start_time)).replace(' ', 'T')
#     try:
#         while True:
#             query_params = {
#                 'TableName': TABLE_NAME,
#                 'KeyConditionExpression': 'PK = :pk AND SK BETWEEN :start_sk AND :end_sk',
#                 'ExpressionAttributeValues': {
#                     ':pk': {'S': f'U#{user_email}'},
#                     ':start_sk': {'S': f'{record_name}Record#{start_time}'},
#                     ':end_sk': {'S': f'{record_name}Record#9999-12-31T23:59:59Z'},
#                 },
#             }
            
#             if last_evaluated_key:
#                 query_params['ExclusiveStartKey'] = last_evaluated_key
                
#             response = dynamodb.query(**query_params)
#             new_items.extend(response['Items'])
#             last_evaluated_key = response.get('LastEvaluatedKey')
#             if not last_evaluated_key:
#                 break
            
#         dynamodb_query_end_time = datetime.now()
#         print(f'{user_email} 사용자의 {record_name} Data 걸린 시간 체크 : {dynamodb_query_end_time - dynamodb_query_start_time}')
    
#         return new_items

#     except ClientError as e:
#         print({e.response['Error']['Message']})
#         return None
    
# def convert_to_utc(dt_str: str) -> pd.Timestamp:
#     """+09:00 시간대가 포함된 문자열을 UTC로 변환."""
#     local_time = parse(dt_str)
#     return local_time.astimezone(timezone.utc)

# def create_df_div(query_json):
#     if len(query_json) == 0:
#         return []

#     if 'HeartRate' in query_json[0]['SK']['S']:
#         return pd.DataFrame({
#             #'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
#             'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'] for x in range(len(query_json))]),
#             'bpm': [int(query_json[x]['recordInfo']['M']['samples']['L'][0]['M']['beatsPerMinute']['N']) for x in range(len(query_json))]
#         })
        
#     if 'Steps' in query_json[0]['SK']['S']:
#         return pd.DataFrame({
#             #'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
#             'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'] for x in range(len(query_json))]),
#             'step': [int(query_json[x]['recordInfo']['M']['count']['N']) for x in range(len(query_json))],
#         })
        
#     if 'TotalCaloriesBurned' in query_json[0]['SK']['S']:
#         return pd.DataFrame({
#             #'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json))]),
#             'ds': pd.to_datetime([query_json[x]['recordInfo']['M']['startTime']['S'] for x in range(len(query_json))]),
#             'calorie': [np.round(float(query_json[x]['recordInfo']['M']['energy']['M']['value']['N']), 3) for x in range(len(query_json))],
#         })
    
#     if 'SleepSession' in query_json[0]['SK']['S']:
#         return pd.DataFrame({
#             #'ds_start': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
#             'ds_start': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['startTime']['S'] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
#             #'ds_end': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'].replace('T', ' ')[:19] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
#             'ds_end': pd.to_datetime([query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['endTime']['S'] for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))]),
#             'sleep': [int(query_json[x]['recordInfo']['M']['stages']['L'][y]['M']['stage']['N']) for x in range(len(query_json)) for y in range(len(query_json[x]['recordInfo']['M']['stages']['L']))],
#         })
    
# def prepare_docs(user_email, df, data_type):

    
#     if data_type == 'calorie':
#         #print('is in calorie Data ----- : ', df, len(df))
#         return [
#             {
#                 'user_email': user_email,
#                 'type': data_type,
#                 'value': float(row[data_type]),
#                 'timestamp': row['ds'],
#             }
#             for row in df.to_dict('records')
#         ]

#     elif data_type == 'sleep':
#         return [
#             {
#                 'user_email': user_email,
#                 'type': data_type,
#                 'value': int(row[data_type]),
#                 'timestamp_start': row['ds_start'],
#                 'timestamp_end': row['ds_end'],
#             }
#             for row in df.to_dict('records')
#         ]

        
#     else: 
        
#         return [
#             {
#                 'user_email': user_email,
#                 'type': data_type,
#                 'value': int(row[data_type]),
#                 'timestamp': row['ds']
#             }
#             for row in df.to_dict('records')
#         ]

        
# async def update_db(user_email, df, collection):
#     if len(df) == 0:
#         return 0
    
#     print('*'*50)
#     print(f'user_email : {user_email}, ,df[:5] : {df[:5]}, ,collection : {collection}')
#     if collection == 'bpm':
#         print('timestamp type : ', type(df[:1]['ds'][0]), '----', df[:1]['ds'][0])
#     print('*'*50)
#     start_doc = time.time()

#     documents = prepare_docs(user_email, df, collection)
    
#     print('#'*50)
#     print(documents[:5])
#     if collection == 'bpm':
#         print('timestamp type :', type(documents[0]['timestamp']), '----', documents[0]['timestamp'])
#     print('#'*50)
    
#     end_doc = time.time()
#     # print(documents)
#     print(f'{user_email} - {collection} prepare_docs 걸린 시간 : {end_doc - start_doc}')

#     is_sleep_collection = collection.startswith('sleep')
    
#     bulk_operations = [
#         UpdateOne(
#             {
#                 'user_email': doc['user_email'],
#                 'timestamp_start' if is_sleep_collection else 'timestamp': doc['timestamp_start' if is_sleep_collection else 'timestamp'],
#             },
#             {'$set': doc},
#             upsert=True
#         ) for doc in documents
#     ]
    
#     print('^'*50)
#     print(bulk_operations[:5])
#     print('^'*50)
    
#     batch_size = 1000  # Reduced batch size for better performance
#     total_operations = len(bulk_operations)
#     total_updated = 0
    
#     start_time = time.time()

#     async def process_batch(batch):
#         try:
#             result = await eval(collection).bulk_write(batch, ordered=False)
#             return result.upserted_count + result.modified_count
#         except BulkWriteError as bwe:
#             print(f"Bulk write error: {bwe.details}")
#             return bwe.details['nUpserted'] + bwe.details['nModified']

#     tasks = [
#         process_batch(bulk_operations[i:i+batch_size])
#         for i in range(0, total_operations, batch_size)
#     ]

#     results = await asyncio.gather(*tasks)
#     total_updated = sum(results)
    
#     end_time = time.time()

    
#     #return total_updated

# async def update_db_div(user_email, df, collection):
#     if len(df) == 0:
#         return 0
#     # prepare_docs() 구조 bpm, step, calorie, step 들어가야
#     print(f'user_email : {user_email}, ,df[:3] : {df[:3]}, ,collection : {collection}')
#     start_doc = time.time()
#     documents = prepare_docs(user_email, df, collection)
#     end_doc = time.time()
#     # print(documents)
#     print(f'{user_email} - {collection} prepare_docs 걸린 시간 : {end_doc - start_doc}')
    
    
#     if collection[:collection.find('_')] == 'sleep':
#         bulk_operations = [
#             UpdateOne(
#                 {
#                     'user_email': doc['user_email'],
#                     'timestamp_start': doc['timestamp_start'],
#                 },
#                 {'$set':doc},
#                 upsert=True
#             ) for doc in documents
#         ]
        
#     else:
#         bulk_operations = [
#             UpdateOne(
#                 {
#                     'user_email': doc['user_email'],
#                     'timestamp': doc['timestamp']
#                 },
#                 {'$set':doc},
#                 upsert=True
#             ) for doc in documents
#         ]
        
#     batch_size = 10000
    
#     total_operations = len(bulk_operations)
    
#     start_time = time.time()
    
#     for i in range(0, total_operations, batch_size):
#         batch = bulk_operations[i:i+batch_size]
#         await eval(collection).bulk_write(batch, ordered=False)
        
#     end_time = time.time()
#     print(f'{user_email} 사용자의 {collection} Data 저장 걸린 시간 --> {end_time - start_doc}')

async def start_date(user_email, collection_name):
    if 'sleep' in collection_name:
        res = await eval(collection_name).find_one({'user_email': user_email}, sort=[('timestamp_start', ASCENDING)])

        if res == None:
            return None
        return res['timestamp_start']
    else:
        res = await eval(collection_name).find_one({'user_email': user_email}, sort=[('timestamp', ASCENDING)])
        print('##############', res)
        return res['timestamp']

@app.get("/get_start_dates/{user_email}")
async def get_start_dates(user_email: str):
    #collections = ['bpm_test3', 'step_test3', 'calorie_test3', 'sleep_test3']
    collections = ['bpm', 'step', 'calorie']
    
    print('----?')
    
    return {"start_date": sorted([await start_date(user_email, collections[x]) for x in range(len(collections))])[0]}
    
# @app.get("/get_save_dates/{user_email}")
# async def get_save_dates(user_email: str):
#     #collections = ['bpm_div', 'step_div', 'calorie_div', 'sleep_div']
#     collections = ['bpm', 'step', 'calorie', 'sleep']
    
#     return {"save_dates": [max(exist_collection_div(user_email, collections))]}

# @app.get("/get_save_dates_div/{user_email}")
# async def get_save_dates_div(user_email: str):
#     #collections = ['bpm_test3', 'step_test3', 'calorie_test3', 'sleep_test3']
#     #collections = ['bpm', 'step', 'calorie', 'sleep']
#     collections = ['bpm', 'step', 'calorie', 'sleep']
    
#     return {"save_dates": [max(await exist_collection_div(user_email, collections))]}

# async def get_bpm_hour_data(user_email, start_date, end_date):

#     cursor = bpm.find({'user_email': user_email, 'timestamp': {'$gte': datetime.fromtimestamp(int(str(start_date)[:-3])), '$lte': datetime.fromtimestamp(int(str(end_date)[:-3]))}})
#     results = []
#     async for document in cursor:
#         results.append(document)
#     return results

# @app.get("/feature_hour_div/{user_email}/{start_date}/{end_date}")
# async def bpm_hour_feature(user_email: str, start_date: str, end_date: str):
#     query = await get_bpm_hour_data(user_email, start_date, end_date)
#     mongo_bpm_df = pd.DataFrame({
#         'ds': [doc['timestamp'] for doc in query],
#         'bpm': [doc['value'] for doc in query]
#     })
   
#     if len(mongo_bpm_df) == 0:
#         return 0
#     mongo_bpm_df['ds'] = pd.to_datetime(mongo_bpm_df['ds'])
#     mongo_bpm_df = mongo_bpm_df.astype({'bpm': 'int32'})

#     hour_group = mongo_bpm_df.groupby(mongo_bpm_df['ds'].dt.floor('h'))
#     hour_hrv = hour_group.apply(calc_hrv).reset_index()
    
#     hour_hrv['ds'] = hour_hrv['ds'].dt.tz_localize('UTC')
    
#     hour_hrv.rename(columns={'rmssd' : 'hour_rmssd', 'sdnn' : 'hour_sdnn'}, inplace=True)
    
#     return {'hour_hrv': hour_hrv[['ds', 'hour_rmssd', 'hour_sdnn']].to_dict('records')}

async def get_bpm_all_data(user_email, start_date):

    cursor = bpm.find({'user_email': user_email, 'timestamp': {'$gte': start_date}})
    results = await cursor.to_list(length=None)
    return results

# async def get_hrv_all_data(user_email):

#     cursor_rmssd = rmssd.find({'user_email': user_email})
#     cursor_sdnn = sdnn.find({'user_email': user_email})
#     results_rmssd = await cursor_rmssd.to_list(length=None)
#     results_sdnn = await cursor_sdnn.to_list(length=None)

#     df_rmssd = pd.DataFrame({
#         'ds': [doc['timestamp'] for doc in results_rmssd],
#         'day_rmssd': [doc['value'] for doc in results_rmssd]
#     })
#     df_sdnn = pd.DataFrame({
#         'ds': [doc['timestamp'] for doc in results_sdnn],
#         'day_sdnn': [doc['value'] for doc in results_sdnn]
#     })

#     df_merged = pd.merge(df_rmssd, df_sdnn, on='ds', how='outer')

#     df_merged = df_merged.sort_values('ds').reset_index(drop=True)
    
#     return df_merged


########################## 저장 방식 변경에 의한 구조 변경 ###########################
async def get_hour_hrv_data(user_email, start_date, end_date):
    cursor_rmssd = hour_rmssd.find({'user_email': user_email, 'timestamp': {'$gte': datetime.fromtimestamp(int(start_date[:-3])), '$lte': datetime.fromtimestamp(int(end_date[:-3]))}})
    results_rmssd = await cursor_rmssd.to_list(length=None)
    
    cursor_sdnn = hour_sdnn.find({'user_email': user_email, 'timestamp': {'$gte': datetime.fromtimestamp(int(start_date[:-3])), '$lte': datetime.fromtimestamp(int(end_date[:-3]))}})
    results_sdnn = await cursor_sdnn.to_list(length=None)
    
    results = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in results_rmssd],
        'hour_rmssd': [doc['value'] for doc in results_rmssd],
        'hour_sdnn': [doc['value'] for doc in results_sdnn]
    })
    results['ds'] = pd.to_datetime(results['ds'])
    results['ds'] = results['ds'].dt.tz_localize('UTC')
    
    print('result : ', results)
    
    return results

async def get_day_hrv_data(user_email):

    cursor_rmssd = day_rmssd.find({'user_email': user_email})
    cursor_sdnn = day_sdnn.find({'user_email': user_email})
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
    df_merged = df_merged.replace({np.nan: None})
    
    print('df_merged : ', df_merged)
    
    return df_merged


@app.get("/feature_hour_div2/{user_email}/{start_date}/{end_date}")
async def bpm_hour_feature2(user_email: str, start_date: str, end_date: str):
    #print(f'{user_email} 들어옴, {datetime.fromtimestamp(int(start_date[:-3]))} ~ {datetime.fromtimestamp(int(end_date[-2]))}')
    # print('in feature hour start_date: ', start_date)
    # print('in feature hour end_date: ', end_date)
    # print('in feature hour types: ', type(start_date), type(end_date))
    hour_hrv = await get_hour_hrv_data(user_email, start_date, end_date)
    
    return {'hour_hrv': hour_hrv[['ds', 'hour_rmssd', 'hour_sdnn']].to_dict('records')}

@app.get("/feature_day_div2/{user_email}")
async def bpm_day_feature2(user_email: str):
    day_hrv = await get_day_hrv_data(user_email)
    print('feature day div2 day_hrv : ', day_hrv)
    return {'day_hrv': day_hrv[['ds', 'day_rmssd', 'day_sdnn']].to_dict('records')}
########################################################################


# # Heatmap 수정본
# @app.get("/feature_day_div/{user_email}")
# async def bpm_day_feature(user_email: str):

#     print('@@@@@@@@@@@@@@@@@@@@@@@ IN FEATURE DAY DIV @@@@@@@@@@@@@@@@@@@@@@@@@@')

#     update_bpm_ds = await exist_collection_div(user_email, ['bpm'])

#     update_bpm_data = new_query_div(user_email, 'HeartRate', update_bpm_ds[0])
#     # print(update_bpm_data)
#     if len(update_bpm_data) > 0:
#         update_df = create_df_div(update_bpm_data)

#         await update_db_div(user_email, update_df, 'bpm')

#     rmssd_last_date = await rmssd.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])

#     bpm_last_date = await bpm.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    
#     print(rmssd_last_date)
    
#     if rmssd_last_date == None:
#         rmssd_last_date = {'timestamp': datetime(1,1,1)}
#     # 업데이트 된 데이터가 있다면, HRV Data 마지막 timestamp <-> BPM 데이터 마지막 timestamp 비교
#     if bpm_last_date['timestamp'] - rmssd_last_date['timestamp'] > timedelta(days=1):
  
#         update_rmssd_query = await bpm.find({'user_email': user_email, 'timestamp': {'$gte': rmssd_last_date['timestamp'], '$lte': bpm_last_date['timestamp']}}).to_list(length=None)
#         update_rmssd_df = pd.DataFrame({
#                 'ds': pd.to_datetime([update_rmssd_query[x]['timestamp'] for x in range(len(update_rmssd_query))]),
#                 'bpm': [int(update_rmssd_query[x]['value']) for x in range(len(update_rmssd_query))]
#             })
        
#         update_rmssd_df.sort_values(by='ds', ascending=True, inplace=True)
#         update_rmssd_df.reset_index(inplace=True, drop=True)
#         update_rmssd_df['ds'] = update_rmssd_df['ds'].dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        
#         day_group = update_rmssd_df.groupby(update_rmssd_df['ds'].dt.floor('d'))
#         day_hrv = day_group.apply(calc_hrv).reset_index()
        
#         day_hrv['ds'] = (day_hrv['ds'] - timedelta(hours=9)).dt.tz_localize(None)
        
#         day_hrv.rename(columns={'rmssd' : 'day_rmssd', 'sdnn' : 'day_sdnn'}, inplace=True)
        
        
        
#         save_hrv(user_email, day_hrv)

#     day_hrv = await get_hrv_all_data(user_email)

#     print('@@@@@@@@@@@@@@@@@@@@@@@ OUT FEATURE DAY DIV @@@@@@@@@@@@@@@@@@@@@@@@@@')
#     return {'day_hrv': day_hrv[['ds', 'day_rmssd', 'day_sdnn']].to_dict('records')}




@app.get("/predict_minute_div/{user_email}")
async def bpm_minute_predict(user_email: str):
    print('&&&&&&&&&&&&&&&&&&&&&&&& IN PREDICT MINUTE DIV &&&&&&&&&&&&&&&&&&&&&&&&')
    start_time = datetime.now()

    last_bpm_data = await bpm.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    query = await get_bpm_all_data(user_email, last_bpm_data['timestamp'] - timedelta(days=15))
    mongo_bpm_df = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in query],
        'bpm': [doc['value'] for doc in query]
    })
    mongo_bpm_df = mongo_bpm_df[mongo_bpm_df['ds'].dt.second == 0]
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    min_model = Prophet(
        changepoint_prior_scale=0.0001,
        seasonality_mode='multiplicative',
    )

    min_model.fit(mongo_bpm_df)
    
    min_future = min_model.make_future_dataframe(periods=60*24*1, freq='min')
    min_forecast = min_model.predict(min_future)
    
    min_forecast.rename(columns={'yhat': 'min_pred_bpm'}, inplace=True)
    min_forecast['min_pred_bpm'] = np.round(min_forecast['min_pred_bpm'], 3)  
    min_forecast = min_forecast[len(min_forecast) - 60*24*1:]
    
    end_time = datetime.now()
    print(f'in predict_minute_div 걸린 시간 : {end_time - start_time} --- {len(mongo_bpm_df)}')

    min_forecast['ds'] = min_forecast['ds'].dt.tz_localize('UTC')
    print('&&&&&&&&&&&&&&&&&&&&&&&& OUT PREDICT MINUTE DIV &&&&&&&&&&&&&&&&&&&&&&&&')
    return {'min_pred_bpm': min_forecast[['ds', 'min_pred_bpm']].to_dict('records')}
       
@app.get("/predict_hour_div/{user_email}")
async def bpm_hour_predict(user_email: str):
    print('$$$$$$$$$$$$$$$$$$$$$$$$ IN PREDICT MINUTE DIV $$$$$$$$$$$$$$$$$$$$$$$$')
    start_time = datetime.now()

    last_bpm_data = await bpm.find_one({'user_email': user_email}, sort=[('timestamp', DESCENDING)])
    query = await get_bpm_all_data(user_email, last_bpm_data['timestamp'] - timedelta(days=30))

    mongo_bpm_df = pd.DataFrame({
        'ds': [doc['timestamp'] for doc in query],
        'bpm': [doc['value'] for doc in query]
    })
    mongo_bpm_df.rename(columns={'bpm': 'y'}, inplace=True)
    hour_df = mongo_bpm_df.groupby(mongo_bpm_df['ds'].dt.floor('h')).agg({'y':'mean'}).reset_index()
    print('$$$$', hour_df[:5])
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
    

    hour_forecast['ds'] = hour_forecast['ds'].dt.tz_localize('UTC')
    
    #hour_forecast.to_csv('hour_forecast.csv')

    print('$$$$$$$$$$$$$$$$$$$$$$$$ OUT PREDICT MINUTE DIV $$$$$$$$$$$$$$$$$$$$$$$$')
    return {'hour_pred_bpm': hour_forecast[['ds', 'hour_pred_bpm']][len(hour_forecast) - 72:].to_dict('records')}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    