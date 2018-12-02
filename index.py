from flask import render_template
from flask import Flask, request, redirect, url_for
from flask_restful import Resource, Api
import sys
import os
from werkzeug.utils import secure_filename
sys.path.insert(0, './backend')
from cos_sim import getCategory

UPLOAD_FOLDER = 'data/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

api = Api(app)

class Category(Resource):
    def get(self, str):
        return {'Category': getCategory(str)}

api.add_resource(Category, '/<string:str>')

