from flask import Flask
from web.routes import web_app

app = Flask(
    __name__,
    static_folder="web/static",           # ← karena kamu pakai web/static
    template_folder="web/templates"       # ← ini yang penting!
)

app.register_blueprint(web_app)

if __name__ == "__main__":
    app.run(debug=True)
