
from pymongo import MongoClient, errors
import streamlit as st


class DatabaseConnection:
    '''This class is to create an object to connect to the database.'''
    @st.cache_resource
    def __init__(self, database_name : str = 'face_id_app', address : str = 'localhost', port : int = 27017):
        try:
            self.client = MongoClient(f'mongodb://{address}:{port}/')
            self.database = self.client[database_name]
        except errors.ConnectionFailure as error:
            print(error)
    @st.cache_data(ttl=600)
    def get_database(self):
        '''This method is to get the database.'''
        return self.database


class DB_Requests:
    @st.cache_data(ttl=600)
    def __init__(self, database_name, address : str = 'localhost', port : int = 27017):
        self.db = DatabaseConnection(database_name).get_database()

    @st.cache_data(ttl=600)
    def get_collection(self, collection_name):
        '''This method is to get a collection from the database.'''
        a = self.db[collection_name]
        return a
    
    @st.cache_data(ttl=600)
    def get_user_embedding(self, collection_name, user_name):
        a = self.db.get_collection(collection_name).find_one({"user_name": user_name})
        

        return a['user_embedding']
    
    @st.cache_data(ttl=600)
    def delete_user(self, collection_name, user_name):
        '''This method is to delete a document from the database.'''
        self.db.get_collection(collection_name).delete_one({"user_name": user_name})


    @st.cache_data(ttl=600)
    def insert_user(self, collection_name, user_name, user_embedding):
        '''This method is to insert a document into the database.'''
        self.db.get_collection(collection_name).insert_one({"user_name": user_name, "user_embedding": user_embedding})

        




#streamlit example
# @st.cache_resource
# def init_connection():
#     return MongoClient(**st.secrets["mongo"])

# client = init_connection()

# # Pull data from the collection.
# # Uses st.cache_data to only rerun when the query changes or after 10 min.
# @st.cache_data(ttl=600)
# def get_data():
#     db = client.mydb
#     items = db.mycollection.find()
#     items = list(items)  # make hashable for st.cache_data
#     return items
