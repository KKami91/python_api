from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

class Database:
    client: Optional[AsyncIOMotorClient] = None
    db = None

    @classmethod
    async def connect_db(cls, mongodb_url: str):
        """데이터베이스 연결을 초기화합니다."""
        if cls.client is None:
            cls.client = AsyncIOMotorClient(mongodb_url,
                maxPoolSize=10,
                minPoolSize=5,
                maxIdleTimeMS=60000,
                connectTimeoutMS=20000,
                retryWrites=True,
                serverSelectionTimeoutMS=10000
            )
            cls.db = cls.client.get_database("heart_rate_db")
            
            # 연결 테스트
            try:
                await cls.client.admin.command('ping')
                print("MongoDB 연결 성공")
            except Exception as e:
                print(f"MongoDB 연결 실패: {e}")
                await cls.close_db()
                raise

    @classmethod
    async def close_db(cls):
        """데이터베이스 연결을 종료합니다."""
        if cls.client is not None:
            await cls.client.close()
            cls.client = None
            cls.db = None
            print("MongoDB 연결 종료")

    @classmethod
    @asynccontextmanager
    async def get_db(cls):
        """데이터베이스 연결의 컨텍스트 매니저."""
        if cls.db is None:
            await cls.connect_db()
        try:
            yield cls.db
        except Exception as e:
            print(f"데이터베이스 작업 중 오류 발생: {e}")
            raise