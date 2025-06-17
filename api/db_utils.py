# api_server/utils/db_utils.py
from sqlalchemy import create_engine
import sqlparse

def get_engine():
    server = 'your_server'
    database = 'your_database'
    username = 'readonly_user'
    password = 'your_password'
    driver = 'ODBC Driver 17 for SQL Server'
    connection_string = (
        f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
    )
    return create_engine(connection_string)

def get_tables(engine):
    sql = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"
    with engine.connect() as conn:
        result = conn.execute(sql)
        return [row[0] for row in result]

def get_columns(engine, table):
    sql = f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}'"
    with engine.connect() as conn:
        result = conn.execute(sql)
        return result.fetchall()

def run_sql(engine, sql):
    with engine.connect() as conn:
        result = conn.execute(sql)
        return result.fetchall(), result.keys()

def is_safe_sql(sql):
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        if stmt.get_type() != 'SELECT':
            return False
    return True