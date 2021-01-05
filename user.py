from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, LoginManager

login = LoginManager()
login.login_view = "login"


class UserModel(UserMixin):
    # __tablename__ = 'users'

    def __init__(self, email, username, password):
        self.id = username
        self.email = email
        self.password = password

    # email = db.Column(db.String, primary_key=True)
    # username = db.Column(db.String)
    # password = db.Column(db.String)


@login.user_loader
def load_user(id):
    return UserModel.query.get(int(id))

