import psycopg2
from config import DB_CONFIG
from app.generate_vector import get_embedding
import reports100  

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Setup table
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
CREATE TABLE IF NOT EXISTS biopsy_reports (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);
""")

# Load all reports dynamically
all_reports = [getattr(reports100, attr) for attr in dir(reports100) if attr.startswith("report")]

# Insert them
for report in all_reports:
    embedding = get_embedding(report)
    embedding_list = embedding  
    cur.execute("""
        INSERT INTO biopsy_reports (content, embedding)
        VALUES (%s, %s);
    """, (report, embedding_list))

conn.commit()
cur.close()
conn.close()

print("✅ All biopsy reports inserted successfully.")
