from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
import logging

from common.report import UserReport, ProtectedReport
from server.database import Database

address = "127.0.0.1"
port = 8080

auth = HTTPBasicAuth()
db = Database()
app = Flask("brain-pick")


@app.route("/")
def index():
    return "Welcome to Brain-Pick"


@app.route("/api/page", methods=["GET"])
def page():
    elements = db.get_all_elements()
    data = [element.asdict() for element in elements]
    return jsonify(data)


@app.route("/api/internal/userdata/<username>", methods=["GET"])
def get_report(username):
    reports = db.get_all_reports(username)
    data = [report.asdict() for report in reports]
    return jsonify(data)


@auth.verify_password
def verify_business_user(username, password):
    return db.check_business_password(username, password)


@app.route("/api/userdata/<username>", methods=["GET"])
@auth.login_required
def get_protected_report(username):
    reports = db.get_all_reports(username)
    protected_reports = [ProtectedReport(topic=r.topic, value=r.value) for r in reports]
    data = [report.asdict() for report in protected_reports]
    return jsonify(data)


@app.route("/api/report", methods=["POST"])
def submit_report():
    data = request.get_json()
    report = UserReport(**data)
    logging.info("Got report for %s at %s", report.username, report.timestamp)
    db.add_report(report)
    return jsonify({"status": "ok"}), 201


if __name__ == "__main__":
    app.run(address, port)
