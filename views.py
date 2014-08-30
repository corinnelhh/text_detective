from flask import Flask
from flask import render_template, request, url_for, redirect

from detective import FortuneTeller


app = Flask(__name__)


@app.route('/')
def show_main_page():
    u"""Displays base.html."""
    prediction = request.args.get('prediction')
    return render_template('base.html', prediction=prediction)


@app.route('/submit', methods=['GET', 'POST'])
def get_input_text():
    u"""Receive input text from site."""
    submission = request.form['submission']
    answer = "Hey, you said {}".format(submission)
    return redirect(url_for('show_main_page', prediction=answer))

if __name__ == '__main__':
    ft = FortuneTeller()
    app.run(debug=True)
    # from wsgiref.simple_server import make_server
    # srv = make_server('localhost', 8000, app)
    # srv.serve_forever()
