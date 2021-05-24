"""

    *Brownian Motion + Monte Carlo learn expected stock's price from given input `symbol`, `start_date`, `end_date`.
    *Note that this method is not used to predict stock's price. This is for learning the uncertainty of the stock in given time frame only!
    *Applied Gaussian noise to volatility.

    *FROM WIKIPEDIA: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    *Geometry Brownian Motion (GBM) is not a completely realistic model, in particular it falls short of reality in the following points:
    - In real stock prices, volatility changes over time (possibly stochastically), but in GBM, volatility is assumed constant.
    - In real life, stock prices often show jumps caused by unpredictable events or news, but in GBM, the path is continuous (no discontinuity).
    - In an attempt to make GBM more realistic as a model for stock prices, one can drop the assumption that the volatility (σ) is constant. If we assume that the volatility is a deterministic function of the stock price and time, this is called a local volatility model. If instead we assume that the volatility has a randomness of its own—often described by a different equation driven by a different Brownian Motion—the model is called a stochastic volatility model.

"""


from visualize_utils import *
from tqdm import tqdm
import yfinance as yf
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=FutureWarning) 
#np.random.seed(1)  # Deterministic seed, just for testing purpose!

INDICATOR = 'Close'

def print_latest_price(symbol):  # Return latest stock price.
    assert symbol is not None
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')  # Fetch data for the latest day only.
    print('--> Open Price: %.2f$, Closed Price: %.2f$' % (todays_data['Open'][0], todays_data['Close'][0]))


def assert_predicted_price(symbol=None, end_date=None, num_predicting_days=1, predicted_price=None, accepted_error_rate=0.03):
    assert end_date is not None and \
            predicted_price is not None and \
            predicted_price >= 0 and \
            num_predicting_days > 0 and \
            symbol is not None
            
    from datetime import datetime, timedelta
    target_date = datetime.strptime(end_date, '%Y-%m-%d').date() + timedelta(days=num_predicting_days)
    today = datetime.today().date()

    if target_date > today:
        return
    else:
        ticker = yf.Ticker(symbol)
        price_history = ticker.history(start=str(target_date), end=str(target_date+timedelta(days=1)))
        if price_history.size == 0:
            price_history = ticker.history(start=str(target_date-timedelta(days=1)), end=str(target_date))

        truth_price = price_history[INDICATOR][-1]
        print('--> Truth Price: %.2f$' % truth_price)
        if np.abs(truth_price-predicted_price)/truth_price <= accepted_error_rate:
            print(f'--> Good estimation with accepted error rate: {accepted_error_rate}')
        else:
            print(f'--> Bad estimation with accepted error rate: {accepted_error_rate}')


def brownian_motion(price_history=None, days=10, num_trials=100):  # Monte Carlo with Brownian Motion..
    assert price_history is not None and days > 0 and num_trials > 0, 'Invalid input'

    # log scale of percentage changes throughout each day
    log_returns = np.log(1+price_history.pct_change())

    mean = log_returns.mean()
    variance = log_returns.var()
    standard_dev = log_returns.std()

    #Z = np.random.normal(loc=0.0, scale=1.0, size=(days, num_trials))  # Standard Norm data
    Z = np.random.normal(loc=mean, scale=standard_dev, size=(days, num_trials))  # Norm data

    drift = mean - (1/2)*variance
    volatility = standard_dev * Z  # Varying volatility based on normal distribution of log returns.
    daily_returns = np.exp(drift + volatility)

    price_paths = np.zeros(daily_returns.shape)
    price_paths[0] = price_history.iloc[-1]

    for day in tqdm(range(1, days)):
        price_paths[day] = price_paths[day-1]*daily_returns[day]
    
    return log_returns, price_paths


""" MAIN """
if __name__ == "__main__":
    ''' PARAMS '''
    symbol = 'GOOG'
    start_date = '2018-01-01'
    end_date = '2021-04-22'
    num_predicting_days = 30
    num_trials = 100000

    # Fetch data from yfinance API.
    # Use stock's CLOSED PRICE as main indicator.
    price_history = yf.Ticker(symbol).history(start=start_date, end=end_date)
    price_close_history = price_history[INDICATOR]

    print(f'--> SYMBOL: {symbol}')
    print(f'--> Running Brownian Motion from {start_date} to {end_date} with {num_trials} trials..')
    log_returns, price_paths = brownian_motion(price_close_history, num_predicting_days, num_trials)

    # Calculate expected price for prediction.
    predicted_price = price_paths[num_predicting_days-1].mean()
    print_latest_price(symbol)
    print('--> Expected Daily Returns: +- %.5f$' % log_returns.std())
    print('--> Expected Price after %d days from %s: %.2f$' % (num_predicting_days, end_date, predicted_price))

    # Test if the predicted price is ok..
    assert_predicted_price(symbol, end_date, num_predicting_days, predicted_price, accepted_error_rate=0.05)

    # Visualize
    #visualize_stock_price(price_close_history, show=True)
    #visualize_log_returns(log_returns, show=True)
    #visualize_brownian_motion(price_paths, show=True)
    #visualize_predicted_prices(num_predicting_days, price_paths, show=True)
