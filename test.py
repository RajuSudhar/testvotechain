import os
import time
import functions
from database import convert_image_to_binary
import mysql.connector
from mysql.connector import Error

path = "Real/"

try:
    # Connect to MySQL
    connection = mysql.connector.connect(
        host='localqhost',
        database='testvotechain',
        user='python',
        password='Impython312'
    )

    if connection.is_connected():
        cursor = connection.cursor()

        termination_count, bifurcation_count = functions.extract_feature_val(path + "")

        fingerprint_image = convert_image_to_binary(path + "")

        select_query = """
                    SELECT * FROM fingerprint_data
                    WHERE bifurcation_count BETWEEN ? AND ?
                    AND termination_count BETWEEN ? AND ?;"""

        # Data tuple
        data_tuple = (fingerprint_image, termination_count, bifurcation_count)

        # Execute the query and commit
        cursor.execute(select_query, data_tuple)
        connection.commit()

except Error as e:
    print(f"Error: {e}")

finally:
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")


print(time.time())



# query for fetching from database
# SELECT * FROM fingerprint_data
# WHERE bifurcation_count BETWEEN ? AND ?
# AND termination_count BETWEEN ? AND ?;