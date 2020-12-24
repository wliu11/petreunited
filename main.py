import io
# import os
#
# from google.cloud import vision
#
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ServiceAccountToken.json'
# client = vision.ImageAnnotatorClient()
#
# path = '/Users/liuwendy/petreunited/pic.jpg'
#
#
# image_uri = 'gs://cloud-samples-data/vision/using_curl/shanghai.jpeg'
#
# image = vision.Image()
#
# image.source.image_uri = image_uri
#
# response = client.label_detection(image)
#
# print('Labels (and confidence score):')
# print('=' * 30)
# for label in response.label_annotations:
#     print(label.description, '(%.2f%%)' % (label.score*100.))


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
import os

from flask import Flask, render_template, url_for, request, flash, redirect
import sqlite3
from flask_sqlalchemy import SQLAlchemy
from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename


def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


app = Flask(__name__)
app.config['SECRET_KEY'] = 'key0'
app.config['IMAGE_UPLOADS'] = '/Users/liuwendy/petreunited/static/css/img/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = ['PNG', 'JPG', 'JPEG']
app.config['MAX_IMAGE_FILESIZE'] = 0.5 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'
db = SQLAlchemy(app)


class FileContents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300))
    data = db.Column(db.LargeBinary)


def allowed_image(filename):
    if filename == "" or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1]
    print("extension is '" + str(extension) + "'")
    if extension.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    return False


def allowed_filesize(filesize):
    print("filesize is ", filesize)
    if int(filesize) <= app.config['MAX_IMAGE_FILESIZE']:
        return True
    return False


@app.route('/')
def hello():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)


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
    return render_template('post.html', post=post)


@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        if request.files:

            if not allowed_filesize(request.cookies.get('filesize')):
                return redirect(request.url)

            image = request.files['image']
            filename = image.filename

            if allowed_image(filename):
                safeFilename = secure_filename(filename)
                image.save(os.path.join(app.config['IMAGE_UPLOADS'], safeFilename))
            else:
                flash("Please upload an image of type JPG, JPEG or PNG.")
                return redirect(request.url)

        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO posts (title, content) VALUES (?, ?)',
                         (title, content))
            conn.commit()
            conn.close()
            return redirect(url_for('hello'))
    return render_template('create.html')


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
            return redirect(url_for('hello'))

    return render_template('edit.html', post=post)


@app.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post['title']))
    return redirect(url_for('hello'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
