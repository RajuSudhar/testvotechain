import os
import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

def connect_mydb() -> PooledMySQLConnection | MySQLConnectionAbstract:
    connection = mysql.connector.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return connection

def insert_fingerprint_data(fingerprint_binary: bytes, termination_count: int, bifurcation_count: int) -> None:
    try:
        # Connect to MySQL
        connection = connect_mydb()

        if connection.is_connected():
            cursor = connection.cursor()
            # SQL query to insert data
            insert_query = """
            INSERT INTO fingerprint_data (fingerprint_image, termination_count, bifurcation_count)
            VALUES (%s, %s, %s)
            """

            # Data tuple
            data_tuple = (fingerprint_binary, termination_count, bifurcation_count)

            # Execute the query and commit
            cursor.execute(insert_query, data_tuple)
            connection.commit()

            print("Fingerprint data inserted successfully")

    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def fetch_fingerprint(fingerprint_id):
    result = None
    try:
        connection = connect_mydb()
        if connection.is_connected():
            cursor = connection.cursor()
            sql_query = '''SELECT fingerprint_image FROM fingerprint_data where id == %s'''
            cursor.execute(sql_query, fingerprint_id)
            result = cursor.fetchall()
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

        return result

