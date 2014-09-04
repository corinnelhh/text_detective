from flask import Flask
from flask import render_template, request, url_for, redirect

from detective import Detective_


app = Flask(__name__)


@app.route('/')
def show_main_page():
    u"""Displays base.html."""
    prediction = request.args.get('prediction')
    with open('texts/about.txt', 'r') as f:
        about = f.read()
    return render_template('base.html', prediction=prediction, about=about)


@app.route('/submit', methods=['GET', 'POST'])
def get_input_text(answer=None):
    u"""Receive input text from site."""
    submission = request.form['submission']
    if len(submission.split()) > 100:
        answer = dt.test_teller(submission)
    elif len(submission.split()):
        answer = "Hey, this isn't Twitter! Give us more to work with!"
    return redirect(url_for('show_main_page', prediction=answer))

if __name__ == '__main__':
    dt = Detective_()
    app.run(debug=True)
    # from wsgiref.simple_server import make_server
    # srv = make_server('localhost', 8000, app)
    # srv.serve_forever()
