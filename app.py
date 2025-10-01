from flask import Flask
from flask_cors import CORS
from routes.ndvi import ndvi_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(ndvi_bp, url_prefix="/api")
    CORS(
        app,
        resources={r"/api/.*": {  # <-- note the .*
            "origins": ["http://localhost:5173", "http://localhost:8081"]
        }},
        supports_credentials=True,                 # only if you need cookies
        methods=["GET", "POST", "OPTIONS"],        # allow preflight + POST
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Length"],
        max_age=86400
    )
    return app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5050)