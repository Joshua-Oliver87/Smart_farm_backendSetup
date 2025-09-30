from flask import Flask
from routes.ndvi import ndvi_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(ndvi_bp, url_prefix="/api")
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)