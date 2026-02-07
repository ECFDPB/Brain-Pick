from flask import Flask, abort, request
import logging

from common.report import UserReport

# TODO: Accept config
address = "127.0.0.1"
port = 8080

app = Flask("brain-pick")


@app.route("/")
def index():
    return "Welcome to Brain-Pick"


@app.route("/api/userdata/<username>", methods=["GET"])
def get_userdata(username):
    # TODO: Database
    abort(501, description=username)


@app.route("/api/report", methods=["POST"])
def submit_report():
    json = request.get_json()
    report = UserReport(**json)
    logging.info("Got report for %s at %s", report.username, report.username)


if __name__ == "__main__":
    app.run(address, port)
