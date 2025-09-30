from pymongo import MongoClient
from django.conf import settings
import os

class MongoDBConnection:
    _client = None
    _db = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            mongodb_uri = os.getenv('MONGODB_URI', settings.MONGODB_URI)
            cls._client = MongoClient(mongodb_uri)
        return cls._client
    
    @classmethod
    def get_database(cls):
        if cls._db is None: 
            client = cls.get_client()
            db_name = os.getenv('MONGODB_DB_NAME', settings.MONGODB_DB_NAME)
            cls._db = client[db_name]
        return cls._db
    
    @classmethod
    def get_collection(cls, collection_name):
        db = cls.get_database()
        return db[collection_name]
    
    @classmethod
    def close_connection(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
