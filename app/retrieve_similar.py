import psycopg2
from app.generate_vector import get_embedding
from config import DB_CONFIG

def get_similar_reports(doctor_prompt: str, top_k=1):
    try:
        embedding = get_embedding(doctor_prompt)
        embedding_vector = embedding  # Already a list

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            SELECT id, content
            FROM biopsy_reports
            ORDER BY embedding <#> %s::vector
            LIMIT %s;
        """, (embedding_vector, top_k))

        similar_reports = cur.fetchall()

        cur.close()
        conn.close()

        return [{"id": r[0], "content": r[1]} for r in similar_reports]

    except Exception as e:
        print(f"❌ Error retrieving similar reports: {e}")
        return []
