import mysql.connector
from mysql.connector import Error

def connect_mydb():
    connection = mysql.connector.connect(
        host='localhost',
        database='testvotechain',
        user='python',
        password='Impython312'
    )
    return connection

def insert_fingerprint_data(img_path: str, termination_count: int, bifurcation_count: int) -> None:
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            database='testvotechain',
            user='python',
            password='Impython312'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            fingerprint_image = convert_image_to_binary(img_path)
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



def convert_image_to_binary(file_path: str) -> bytes:
    # Convert the image file to binary data.
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return binary_data