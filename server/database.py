import sqlite3
from datetime import datetime

from common.report import UserReport
from common.page import Tag, Element


class Database:
    def __init__(self, db_path="test.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = 1")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, remark TEXT
        );
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT, width REAL, height REAL, x REAL, y REAL
        );
        CREATE TABLE IF NOT EXISTS tag_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            value REAL CHECK(value >= -1.0 AND value <= 1.0),
            FOREIGN KEY(username) REFERENCES users(username)
        );
        CREATE TABLE IF NOT EXISTS report_tags (
            report_id INTEGER NOT NULL, tag_id INTEGER NOT NULL,
            PRIMARY KEY (report_id, tag_id),
            FOREIGN KEY(report_id) REFERENCES tag_reports(id),
            FOREIGN KEY(tag_id) REFERENCES tags(id)
        );
        CREATE TABLE IF NOT EXISTS element_tags (
            element_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY (element_id, tag_id),
            FOREIGN KEY(element_id) REFERENCES elements(id) ON DELETE CASCADE,
            FOREIGN KEY(tag_id) REFERENCES tags(id)
        );
        CREATE TABLE IF NOT EXISTS business_users (
            username TEXT PRIMARY KEY, token TEXT
        );
        """)
        self._conn.commit()

    def add_user(self, username, remark=""):
        try:
            self._conn.execute("INSERT INTO users (username, remark) VALUES (?, ?)", (username, remark))
            self._conn.commit()
        except sqlite3.IntegrityError:
            print(f"User {username} already exists.")

    def add_report(self, report: UserReport):
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO tag_reports (username, timestamp, value) VALUES (?, ?, ?)",
            (report.username, report.timestamp, report.value),
        )
        report_id = cursor.lastrowid

        for tag_name in report.topic:
            cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
            tag_id = cursor.fetchone()[0]
            cursor.execute(
                "INSERT INTO report_tags (report_id, tag_id) VALUES (?, ?)",
                (report_id, tag_id),
            )

        self._conn.commit()
        print(f"Report {report_id} created for {report.username} with tags {report.topic}")

    def get_all_elements(self):
        cursor = self._conn.cursor()
        cursor.execute("SELECT id, width, height, x, y FROM elements")
        rows = cursor.fetchall()
        return [Element(id=row["id"], width=row["width"], height=row["height"], x=row["x"], y=row["y"], tags=[]) for row in rows]

    def get_all_reports(self, username):
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT
                tr.id AS report_id,
                tr.username,
                tr.timestamp,
                tr.value,
                t.id AS tag_id,
                t.name AS tag_name
            FROM tag_reports tr
            LEFT JOIN report_tags rt ON tr.id = rt.report_id
            LEFT JOIN tags t ON rt.tag_id = t.id
            WHERE tr.username = ?
            ORDER BY tr.timestamp DESC
        """, (username,))
        rows = cursor.fetchall()

        reports_map = {}
        for row in rows:
            r_id = row["report_id"]
            if r_id not in reports_map:
                try:
                    dt_obj = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                    ts_int = int(dt_obj.timestamp())
                except (ValueError, TypeError):
                    ts_int = 0

                reports_map[r_id] = UserReport(
                    username=row["username"],
                    timestamp=ts_int,
                    value=row["value"],
                    topic=[],
                )

            if row["tag_id"] is not None:
                reports_map[r_id].topic.append(row["tag_name"])

        return list(reports_map.values())

    def check_business_password(self, username, password):
        cursor = self._conn.cursor()
        cursor.execute("SELECT token FROM business_users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row and row["token"] == password:
            return username
        return None
