# app/db_utils.py
import psycopg2
from config import DB_CONFIG

# Function to get a DB connection
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)
