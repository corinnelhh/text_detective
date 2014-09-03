from flask import Flask
from flask import render_template, request, url_for, redirect

from detective import Detective_


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
    if len(submission.split()) > 50:
        answer = dt.test_teller(submission)
    elif len(submission.split()):
        answer = "Hey, this isn't Twitter! Give us more to work with!"
    else:
        answer = None
    return redirect(url_for('show_main_page', prediction=answer))

if __name__ == '__main__':
    dt = Detective_()
    # app.run(debug=True)
    app.run(debug=False)
