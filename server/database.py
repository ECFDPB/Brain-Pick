import sqlite3

from common.report import TagReport


class Database:
    # TODO: Stub
    def __init__(self, db_path="my_project.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = 1")
        self.create_tables()

    def create_tables(self):
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, remark TEXT
        );
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tag_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            value REAL CHECK(score >= -1.0 AND score <= 1.0),
            FOREIGN KEY(username) REFERENCES users(username)
        );
        CREATE TABLE IF NOT EXISTS report_tags (
            report_id INTEGER NOT NULL, tag_id INTEGER NOT NULL,
            PRIMARY KEY (report_id, tag_id),
            FOREIGN KEY(report_id) REFERENCES tag_reports(id),
            FOREIGN KEY(tag_id) REFERENCES tags(id)
        );
        """
        self.conn.executescript(schema)
        self.conn.commit()

    def add_user(self, username, extra=""):
        try:
            self.conn.execute(
                "INSERT INTO users (username, extra_data) VALUES (?, ?)",
                (username, extra),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            print(f"User {username} already exists.")

    def add_report(self, report: TagReport):
        username = report.username
        timestamp = report.timestamp
        tags = report.topic
        value = report.value
        cursor = self.conn.cursor()

        cursor.execute(
            "INSERT INTO tag_reports (username, timestamp, score) VALUES (?, ?, ?)",
            (username, timestamp, value),
        )
        report_id = cursor.lastrowid
        for tag_name in tags:
            cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            tag_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO report_tags (report_id, tag_id) VALUES (?, ?)",
                (report_id, tag_id),
            )

        self.conn.commit()
        print(f"Report {report_id} created for {username} with tags {tags}")
