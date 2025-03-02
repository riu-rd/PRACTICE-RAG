from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
import pandas as pd

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def get_stock_signal(symbol: str, interval: str) -> str:
    """Retrieves intraday stock data for the given symbol using the Alpha Vantage API, computes exponential moving averages (EMA), and generates a trading signal based on an EMA crossover strategy.
    Args:
        symbol: A string representing the stock symbol to analyze (e.g., "AAPL", "GOOG", "MSFT", "TSLA").
        interval: A string representing the time interval between data points (e.g., "1min", "5min", "15min", "60min").
    """
    API_KEY = 'RG9XKRIYBL2EV3V3'
    
    url = (
        f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
        f'&symbol={symbol}&interval={interval}&outputsize=full&apikey={API_KEY}'
    )
    
    response = requests.get(url)
    data = response.json()
    
    time_series_key = f"Time Series ({interval})"
    time_series = data.get(time_series_key, {})
    if not time_series:
        raise ValueError("Failed to retrieve data. Check your API key, symbol, and interval.")
    
    # Create a DataFrame from the time series data
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df['close'] = pd.to_numeric(df['close'])
    
    # Ensure there is enough data to calculate the EMAs
    if len(df) < 26:
        return f"Insufficient data to calculate the required EMAs for {symbol} at a {interval} interval."
    
    # Improved Strategy: Using Exponential Moving Averages (EMA) for a more responsive indicator.
    # Short-term EMA (e.g., 12 periods) and Long-term EMA (e.g., 26 periods)
    df['EMA_short'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Get the latest two data points to check for a crossover
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Determine buy/sell signal based on EMA crossover
    signal = "Hold"
    if prev['EMA_short'] < prev['EMA_long'] and latest['EMA_short'] > latest['EMA_long']:
        signal = "Buy"
    elif prev['EMA_short'] > prev['EMA_long'] and latest['EMA_short'] < latest['EMA_long']:
        signal = "Sell"
    
    # Return a complete sentence with the decision
    decision = (f"The latest closing price for {symbol.upper()} at a {interval} interval is "
                f"${latest['close']:.2f}, and the recommended action is to {signal}.")
    return decision

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
# model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_stock_signal, image_generation_tool, get_current_time_in_timezone], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()