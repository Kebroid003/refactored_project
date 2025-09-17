import logging
import sqlite3


def create_tables(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            data_hash TEXT NOT NULL,
            row_count INTEGER,
            file_size INTEGER,
            processing_status TEXT DEFAULT 'pending'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_data_id INTEGER,
            processed_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            transformation_type TEXT,
            output_file TEXT,
            quality_score REAL,
            error_count INTEGER DEFAULT 0,
            FOREIGN KEY (raw_data_id) REFERENCES raw_data (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            generated_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT,
            recipient_email TEXT,
            status TEXT DEFAULT 'generated'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            user_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            ip_address TEXT
        )
    ''')
    connection.commit()
    logging.getLogger(__name__).info("Database tables ensured")


def insert_raw_data_log(connection: sqlite3.Connection, source_file: str, data_hash: str, row_count: int, file_size: int) -> None:
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO raw_data (source_file, data_hash, row_count, file_size)
        VALUES (?, ?, ?, ?)
    ''', (source_file, data_hash, row_count, file_size))
    connection.commit()


def insert_report_log(connection: sqlite3.Connection, report_type: str, file_path: str, status: str = 'completed') -> None:
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO reports (report_type, file_path, status)
        VALUES (?, ?, ?)
    ''', (report_type, file_path, status))
    connection.commit()


