# StockBot

## Technical Documentation
#### To use the bot locally
1. Install Python 3.6 or above. Install Python [here](https://www.python.org/).

2. If using git, run ```git clone https://github.com/zhuodannychen/StockBot```
3. Run ```pip3 install -r requirements.txt``` to install dependencies.
4. To configure the ```DISCORD_TOKEN```, there are two options: Create a ```.env``` in root and set ```DISCORD_TOKEN='YOUR TOKEN'```, or configure it in ```CONFIG.py``` and edit the comments in ```bot.py```.
5. Edit the other TOKENS with your own tokens in ```CONFIG.py```.
6. Run the bot with ```python3 bot.py```.
#### To invite the bot to a server
7. To add the bot to your server, go to this link and follow instructions. <https://discord.com/oauth2/authorize?client_id=759616214536028200&permissions=0&scope=bot>

#### How the Bot Works
For getting real time data, like the !price command, !triple, and !profile, the data is scrapped from [Yahoo Finance](https://finance.yahoo.com/) using [BeautifulSoup](https://pypi.org/project/beautifulsoup4/).

For commands like !news and !snews, the [newsapi](https://newsapi.org/docs/client-libraries/python) is used to get headline news.

For sending news everyday at 7:00 am CDT, I used asyncio and datetime libraries to calculate the time until 7:00 am, then I called on commands of !news and !price.

For the forecast command, a LSTM model created with [Tensorflow](https://www.tensorflow.org/) was used to forecast the stock prices. The data comes from [alpha-vantage API](https://www.alphavantage.co/documentation/). alpha-vantage provides all historical data in [pandas](https://pandas.pydata.org/) format. However, I am only using the past 1200 days because the model trains quicker with a smaller data set. We also used an adam optimizer with mean squared error for loss.

## User Documentation
**!** prefix

**chart** - Returns a tradingview link of a given stock symbol. Tradingview provides advanced charts and indicators for trading.

--- Example: !chart tsla

**news** - Returns (usually 10) headline news from a given source. Default is bbc-news. Here are a list of credible sources:
* business-insider
* cbc-news
* cnn
* fortune
* google-news
* hacker-news
* nbc-news
* reuters

--- Example: !news

--- Example: !news cnn

**snews** -  Returns the headline news of a given stock symbol.

--- Example: !snews msft

**price** -  Returns the current price and percent change of a given stock.

--- Example: !price aapl

**profile** - Returns the profile/about info of a given stock.

--- Example: !profile AMZN

**triple** - Returns the current price and percent change of the three major index (NASDAQ, DJI, and S&P 500).

--- Example: !triple

**forecast** - Returns the forecast prices of the next 5 days, from top to bottom, for a given stock symbol. Note: this command can take up to 60 seconds before returning results and is not very accurate.

--- Example: !forecast msft

If no stock symbol is inputted for any commands, MSFT will be used by default.

In addition, the bot sends news headlines and the premarket movement of NASDAQ 100 everyday at 7:00 am CDT.

## Improvements
* The current stock forecasting model is not very accurate, so a better model can be developed.
* The current forecasting command takes a long time to execute because I am training it on the command. Saving the model after training and then predicting could lead to a better result and faster execution. The only problem is we don't want to overfitt the data.
* The current training data does not provide adjusted open, adjusted high, and adjusted low, so stocks with recent splits cannot be accurately predicted.
* Matplotlib plots cannot be sent over discord messages, so plotting the day or week trend of a stock cannot be created using a command. Implementation for directly displaying charts on discord messages is still needed.
* The time it sends news is 1 hour late (8:00 am) during daylight saving time, the bot can be improved to adjust accordingly.
* More commands can be implemented, such as earnings data, analyst recommendations, etc.
