from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

pipe = pickle.load(open('model/model.pkl', 'rb'))

teams = sorted([
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
])

cities = sorted([
    'Ahmedabad',
    'Bangalore',
    'Chandigarh',
    'Chennai',
    'Delhi',
    'Hyderabad',
    'Jaipur',
    'Kolkata',
    'Lucknow',
    'Mumbai',
    'Pune'
])

@app.route('/')
def home():
    return render_template(
        'index.html',
        teams=teams,
        cities=cities
    )

@app.route('/predict', methods=['POST'])
def predict():

    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']

    target = int(request.form['target'])
    score = int(request.form['score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

    runs_left = target - score
    balls_left = 120 - int(overs * 6)

    wickets_left = 10 - wickets

    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_score': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)

    win = round(result[0][1] * 100, 2)
    lose = round(result[0][0] * 100, 2)

    predicted_winner = batting_team if win > lose else bowling_team

    return render_template(
        'index.html',
        teams=teams,
        cities=cities,
        win=win,
        lose=lose,
        batting_team=batting_team,
        bowling_team=bowling_team,
        predicted_winner=predicted_winner
    )

if __name__ == '__main__':
    app.run(debug=True)