from flask import Flask
from web.routes import web_bp

app = Flask(__name__)
app.secret_key = 'labai_slapta_reiksme'
app.register_blueprint(web_bp)

if __name__ == '__main__':
    app.run(debug=True)
