# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
import os
from google.cloud import vision
import pyrebase
# import flask_resize
from flask import Flask, render_template, url_for, request, flash, redirect
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
import csv


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


app = Flask(__name__)
app.config['SECRET_KEY'] = 'key0'
app.config['IMAGE_UPLOADS'] = '/Users/liuwendy/petreunited/static/css/img/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['PNG', 'JPG', 'JPEG']
app.config['MAX_IMAGE_FILESIZE'] = 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'

db = SQLAlchemy(app)

file = open(r'dog_breeds.csv')
data_csv = csv.reader(file)
breeds = list(data_csv)


config = {
    "apiKey": os.environ.get("PUBLIC_KEY"),
    "authDomain": "pet-reunited-47c0b.firebaseapp.com",
    "databaseURL": "https://pet-reunited-47c0b-default-rtdb.firebaseio.com/",
    "projectId": "pet-reunited-47c0b",
    "storageBucket": "pet-reunited-47c0b.appspot.com",
    "messagingSenderId": "60196556896",
    "appId": "1:60196556896:web:4080100f7c37e323ab3305",
    "measurementId": "G-6Z0YRP2GTY"
  }

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
firebase_db = firebase.database()


class FileContents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


def allowed_image(filename):
    if filename == "" or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1]
    if extension.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    return False


def allowed_filesize(filesize):
    print("filesize is ", filesize)
    if int(filesize) <= app.config['MAX_IMAGE_FILESIZE']:
        return True
    return False


@app.route('/')
def run():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()

    return render_template('index.html', posts=posts)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    file = request.files['inputFile']
    newFile = FileContents(name=file.filename, data=file.read())
    db.session.add(newFile)
    db.session.commit()
    return 'Saved ' + file.filename + ' to the database'


def get_post(post_id):
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?',
                        (post_id,)).fetchone()
    conn.close()
    if post is None:
        abort(404)
    return post


@app.route('/<int:post_id>')
def post(post_id):
    post = get_post(post_id)
    url = post['image']
    return render_template('post.html', post=post, url=url)


def display_image(post_id):
    post = get_post(post_id)
    path = post['image']
    url = storage.child(path).get_url(None)
    print("url is ", url)
    return url


@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image = request.files['image']

        filename = image.filename

        if not allowed_filesize(request.cookies.get('filesize')):
            flash("File size is too large, resizing")
            return redirect(request.url)

        if not allowed_image(filename):
            flash("Please upload an image of type JPG, JPEG or PNG.")
            return redirect(request.url)

        safeFilename = secure_filename(filename)

        path = "images/" + safeFilename

        storage.child(path).put(image)

        if not title:
            flash('Please include a brief title')
        if not content:
            flash('Please include a brief description')
        else:
            conn = get_db_connection()
            url = storage.child(path).get_url(None)
            conn.execute('INSERT INTO posts (title, content, image) VALUES (?, ?, ?)',
                         (title, content, url))
            conn.commit()
            conn.close()
            url = storage.child(path).get_url(None)
            results = run_api(url)

            resulting_breed = get_breed(results)
            return render_template('results.html', results=resulting_breed, url=url)

    return render_template('create.html')


def get_breed(results):
    resulting_breed = []
    print("results were ", results)
    for result in results:
        for breed in breeds:
            if result.upper() == breed[0].upper():
                resulting_breed.append(breed)
    return resulting_breed


@app.route('/<int:id>/edit', methods=('GET', 'POST'))
def edit(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE posts SET title = ?, content = ?'
                         ' WHERE id = ?',
                         (title, content, id))
            conn.commit()
            conn.close()
            return redirect(url_for('run'))

    return render_template('edit.html', post=post)


@app.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post['title']))
    return redirect(url_for('run'))


def run_api(url):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ServiceAccountToken.json'
    client = vision.ImageAnnotatorClient()
    image = vision.Image()
    image.source.image_uri = url
    response = client.label_detection(image)
    ret = []
    for label in response.label_annotations:
        ret.append(label.description)
    return ret


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
