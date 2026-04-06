import sqlite3
import pandas as pd

def load_flipkart_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    data = []
    for table in tables:
        table_name = table[0]
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            data.append(df)
        except:
            continue

    conn.close()

    if len(data) == 0:
        raise ValueError("No valid tables found")

    df = pd.concat(data, ignore_index=True)
    return df
