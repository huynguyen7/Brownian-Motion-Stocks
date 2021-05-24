import matplotlib.pyplot as plt
import seaborn as sns


def visualize_stock_price(price_history=None, show=False, save=False):
    assert price_history is not None, 'Invalid input'

    price_history.plot(figsize=(15,6), grid=True, legend=True)
    plt.title('Stock Price')
    if show:
        plt.show()
    if save:
        plt.savefig('stock_price.png')
        plt.close()


def visualize_log_returns(log_returns=None, num_bins=10, show=False, save=False):
    assert log_returns is not None, 'Invalid input'

    plt.grid()
    plt.hist(log_returns, bins=num_bins)
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.title('LOG RETURNS')

    if show:
        plt.show()
    if save:
        plt.savefig('log_returns.png')
        plt.close()


def visualize_brownian_motion(price_paths=None, show=False, save=False):
    assert price_paths is not None, 'Invalid input.'
    plt.xlabel('Day (Indexed 0)')
    plt.ylabel('Price')
    plt.grid()
    plt.plot(price_paths)
    plt.title('PRICE PATHS')
    if show:
        plt.show()
    if save:
        plt.savefig('price_paths.png')
        plt.close()


def visualize_predicted_prices(num_predicting_days=1, price_paths=None, show=False, save=False):
    assert price_paths is not None
    
    sns.distplot(price_paths[num_predicting_days-1])
    plt.scatter(x=[price_paths[num_predicting_days-1].mean()], y=[0.0], c='Red')
    plt.grid()
    plt.xlabel(f'Price after {num_predicting_days} days')
    plt.ylabel('Probability')
    plt.title('PRICE DISTRIBUTION')

    if show:
        plt.show()
    if save:
        plt.savefig('predicted_price.png')
        plt.close()
