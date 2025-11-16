
from flask import Blueprint

controller = Blueprint("api", __name__, url_prefix="/api/rest")

@controller.route('/')
def hello():
    return 'Hello World'

@controller.route('/<section>')
def hello2(section):
    return section+'n'

@controller.route('/test')
def tests():
    return 'etst\n'