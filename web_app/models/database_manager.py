import mysql.connector
from mysql.connector import Error

class DatabaseManager:
    def __init__(self, host="localhost", user="root", password="123456", database="weed_detection"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self._init_db()

    def get_connection(self):
        try:
            return mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except Error as e:
            print(f"Error connecting to MySQL Database: {e}")
            return None

    def _init_db(self):
        # Create database and table if not exists
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            if conn.is_connected():
                cursor = conn.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                cursor.execute(f"USE {self.database}")
                
                create_table_query = """
                CREATE TABLE IF NOT EXISTS detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    image_path VARCHAR(255),
                    weed_density FLOAT,
                    young_density FLOAT,
                    mature_density FLOAT,
                    weed_count INT,
                    spray_time INT,
                    blur_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
                cursor.execute(create_table_query)
                conn.commit()
                cursor.close()
                conn.close()
        except Error as e:
            print(f"Error initializing Database: {e}")

    def save_detection(self, data):
        """
        data expected to be a dictionary:
        {
            'image_path': str,
            'weed_density': float,
            'young_density': float,
            'mature_density': float,
            'weed_count': int,
            'spray_time': int,
            'blur_score': float
        }
        """
        conn = self.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO detections 
                (image_path, weed_density, young_density, mature_density, weed_count, spray_time, blur_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                data.get('image_path', ''),
                data.get('weed_density', 0.0),
                data.get('young_density', 0.0),
                data.get('mature_density', 0.0),
                data.get('weed_count', 0),
                data.get('spray_time', 0),
                data.get('blur_score', 0.0)
            )
            cursor.execute(query, values)
            conn.commit()
            return True
        except Error as e:
            print(f"Failed to insert record into detections table: {e}")
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_history(self, limit=50):
        conn = self.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            query = "SELECT * FROM detections ORDER BY created_at DESC LIMIT %s"
            cursor.execute(query, (limit,))
            records = cursor.fetchall()
            return records
        except Error as e:
            print(f"Failed to read records from detections table: {e}")
            return []
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
