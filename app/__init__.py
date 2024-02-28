from flask import Flask
from app.routes.webAPI import WebQA_blueprint

def create_app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(WebQA_blueprint)

    return flask_app