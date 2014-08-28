from flask import Flask
from flask import render_template
app = Flask(__name__)


@app.route('/')
def show_main_page():
    u"""Displays base.html."""
    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)
    # from wsgiref.simple_server import make_server
    # srv = make_server('localhost', 8000, app)
    # srv.serve_forever()
