from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()

def insert_report(text):
    embedding = get_embedding(text)
    conn = psycopg2.connect(dbname="your_db", user="your_user", password="your_password", host="localhost")
    cur = conn.cursor()
    cur.execute("INSERT INTO biopsy_reports (content, embedding) VALUES (%s, %s)", (text, embedding))
    conn.commit()
    cur.close()
    conn.close()

def find_similar_reports(query_text, top_k=3):
    query_vector = get_embedding(query_text)
    conn = psycopg2.connect(dbname="your_db", user="your_user", password="your_password", host="localhost")
    cur = conn.cursor()
    cur.execute("""
        SELECT content FROM biopsy_reports
        ORDER BY embedding <-> %s
        LIMIT %s
    """, (query_vector, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in results]
