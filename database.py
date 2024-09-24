import mysql.connector
from mysql.connector import Error


def insert_fingerprint_data(fingerprint_image: bytes, termination_count: int, bifurcation_count: int) -> None:
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            database='your_database_name',
            user='your_username',
            password='your_password'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # SQL query to insert data
            insert_query = """
            INSERT INTO fingerprint_data (fingerprint_image, termination_count, bifurcation_count)
            VALUES (%s, %s, %s)
            """

            # Data tuple
            data_tuple = (fingerprint_image, termination_count, bifurcation_count)

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
