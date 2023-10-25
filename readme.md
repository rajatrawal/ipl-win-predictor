# IPL Win Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20Predictor-brightgreen)](https://ipl-match-predictor.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub%20Repo-IPL%20Win%20Predictor-green)](https://github.com/rajatrawal/ipl-win-predictor)
[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Model-blue)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[![NumPy](https://img.shields.io/badge/NumPy-1.19-blue)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-1.2-blue)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-0.80-blue)](https://www.streamlit.io/)

Welcome to the "IPL Win Predictor" project! This machine learning model, built using logistic regression, predicts the probability of a team winning an IPL match based on the current match situation. Get ready to make data-driven predictions!

## About This Project

The "IPL Win Predictor" leverages logistic regression to provide insights into the probability of a team winning an IPL match. This model analyzes various match features, team performance, and player statistics to offer real-time predictions.

## Project Preview
![Capture](https://github.com/rajatrawal/ipl-win-predictor/assets/72153827/071a020f-0bf5-4872-904f-5f9a0e928fd1)


## Explore the Project

[![Live Demo](https://img.shields.io/badge/Live%20Demo-View%20Predictor-brightgreen)](https://ipl-match-predictor.streamlit.app/)

### Features

- **Real-Time Predictions**: Get live predictions for IPL match outcomes based on the current match situation.

- **Interactive Interface**: The predictor is deployed on Streamlit, offering a user-friendly interface for exploring match scenarios.

- **Customizable Inputs**: Adjust the match parameters and teams to simulate different match scenarios.

- **Deployment**: Hosted on Streamlit Cloud for easy access and sharing.

## Usage

To make predictions, provide the following parameters when prompted:

- **Batting Team**: The team currently at bat.
- **Bowling Team**: The team currently bowling.
- **City**: The location of the match.
- **Current runs**: The current score of batting team.
- **Overs Completed**: The number of overs completed.
- **Wickets**: The number of wickets lost.
- **Target Runs**: The total runs scored by a bowling team.

The predictor will calculate the probability of the batting team winning based on these parameters and the current match situation.


## Technologies Used

This project leverages the following technologies:

- [Python](https://www.python.org/)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [Streamlit](https://www.streamlit.io/)

## Installation

To run this project locally, follow these steps:

1. Clone the repository to your local machine using this command:

   ```shell
   git clone https://github.com/rajatrawal/ipl-win-predictor.git
   ```

2. Navigate to the project directory:

   ```shell
   cd ipl-win-predictor
   ```

3. Install the required Python libraries:

   ```shell
   pip install -r requirements.txt
   ```

4. Run the Streamlit app locally:

   ```shell
   streamlit run app.py
   ```

5. Open the provided local URL in your web browser to access the IPL Win Predictor.

## Usage

To make predictions, provide the current match situation including team performance, player statistics, and match conditions. The predictor will calculate the probability of a team winning.


## Predict with Confidence

Explore the "IPL Win Predictor" and make data-driven predictions about IPL match outcomes. Get real-time insights and enhance your understanding of match dynamics. Visit the [Live Demo](https://ipl-match-predictor.streamlit.app/) and elevate your cricket analysis.

## Contribute

If you'd like to contribute to this project or have suggestions for improvement, please feel free to submit issues or pull requests on [GitHub](https://github.com/rajatrawal/ipl-win-predictor).

Thank you for exploring the "IPL Win Predictor"! We hope this tool assists your IPL match predictions. 🏏🌟
