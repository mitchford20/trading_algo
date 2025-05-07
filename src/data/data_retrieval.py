import os
import json
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

class StockDataRetriever:
    def __init__(self, api_key=None, config_file=None, api_provider="alphavantage"):
        """
        Initialize the data retriever with an API key.
        
        Args:
            api_key (str, optional): Direct API key input
            config_file (str, optional): Path to config file containing API key
            api_provider (str, optional): Which API provider to use ('alphavantage' or 'polygon')
        """
        self.api_provider = api_provider.lower()
        self.api_key = self._get_api_key(api_key, config_file)
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly, via config file, or set STOCK_API_KEY environment variable.")
        
        # Set up API configuration based on provider
        if self.api_provider == "alphavantage":
            self.base_url = "https://www.alphavantage.co/query"
            self.request_interval = 12  # seconds between requests (5 per minute)
        elif self.api_provider == "polygon":
            self.base_url = "https://api.polygon.io"
            self.request_interval = 0.2  # Polygon.io allows 5 requests per second on basic tier
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}. Use 'alphavantage' or 'polygon'")
            
        self.last_request_time = 0
    
    def _get_api_key(self, api_key=None, config_file=None):
        """
        Get API key from various sources in order of precedence:
        1. Direct parameter
        2. Config file
        3. Environment variable
        
        Args:
            api_key (str, optional): Direct API key input
            config_file (str, optional): Path to config file
            
        Returns:
            str: API key if found, None otherwise
        """
        # 1. Check direct parameter
        if api_key:
            return api_key
            
        # 2. Check config file
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        return config['api_key']
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass
                
        # 3. Try default config locations
        default_config_paths = [
            'config.json',
            'src/config.json',
            'src/data/config.json',
            str(Path.home() / '.trading_algo' / 'config.json')
        ]
        
        for path in default_config_paths:
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                    if self.api_provider in config and 'api_key' in config[self.api_provider]:
                        return config[self.api_provider]['api_key']
                    elif 'api_key' in config:
                        return config['api_key']
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
                
        # 4. Check environment variable
        env_var = f"{self.api_provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
            
        # 5. Check generic environment variable
        return os.environ.get('STOCK_API_KEY')
    
    def save_api_key(self, api_key, config_file='src/data/config.json'):
        """
        Save API key to config file.
        
        Args:
            api_key (str): API key to save
            config_file (str): Path to config file
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Read existing config if it exists
        config = {}
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
            
        # Update config - store by provider
        if self.api_provider not in config:
            config[self.api_provider] = {}
        config[self.api_provider]['api_key'] = api_key
        
        # Save config
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.api_key = api_key
            return True
        except Exception as e:
            print(f"Error saving API key: {e}")
            return False
    
    def _throttle_api_calls(self):
        """Throttle API calls to avoid hitting rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            wait_time = self.request_interval - time_since_last_request
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_intraday_stock_data(self, symbol, interval="1min", outputsize="full"):
        """
        Retrieve intraday stock data for a given symbol.
        
        Args:
            symbol (str): The stock symbol (e.g., 'AAPL' for Apple)
            interval (str): Time interval between data points. 
                For AlphaVantage: '1min', '5min', '15min', '30min', '60min'
                For Polygon: '1', '5', '15', '30', '60' (in minutes)
            outputsize (str): 'compact' for latest 100 data points, 'full' for extended data
            
        Returns:
            pandas.DataFrame: A DataFrame containing the intraday stock data
        """
        self._throttle_api_calls()
        
        if self.api_provider == "alphavantage":
            return self._alphavantage_intraday(symbol, interval, outputsize)
        elif self.api_provider == "polygon":
            return self._polygon_intraday(symbol, interval)
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def _alphavantage_intraday(self, symbol, interval="1min", outputsize="full"):
        """Alpha Vantage implementation of intraday data retrieval"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Check if there's an error message in the response
        if "Error Message" in data:
            raise Exception(f"API returned an error: {data['Error Message']}")
        
        # Extract time series data
        time_series_key = f"Time Series ({interval})"
        time_series = data.get(time_series_key)
        if not time_series:
            raise Exception("No time series data found in the API response")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Rename columns to more friendly names
        df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        }, inplace=True)
        
        # Convert string values to float
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Add datetime as a column
        df.index = pd.to_datetime(df.index)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "datetime"}, inplace=True)
        
        return df
    
    def _polygon_intraday(self, symbol, interval="1"):
        """Polygon.io implementation of intraday data retrieval"""
        # Convert interval format from "1min" to "1" if needed
        timespan = interval.replace("min", "")
        
        # Calculate start and end dates (Polygon requires specific range)
        # Default to 2 weeks of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        # Format dates as ISO strings
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Construct URL
        url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/{timespan}/minute/{start_str}/{end_str}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Check if request was successful
        if data.get('status') == 'ERROR':
            raise Exception(f"API returned an error: {data.get('error')}")
        
        # Extract results
        results = data.get('results', [])
        if not results:
            raise Exception("No data found in the API response")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Print some debug information
        print(f"Total data points received: {len(df)}")
        print(f"Date range: {df['t'].min()} to {df['t'].max()}")
        
        # Rename columns to match our standard format
        df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "t": "timestamp",
            "n": "transactions"
        }, inplace=True)
        
        # Convert timestamp (milliseconds) to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by datetime to ensure chronological order
        df = df.sort_values('datetime')
        
        # Print sample of data points
        print("\nSample of data points:")
        print(df[['datetime', 'open', 'close', 'volume']].head())
        
        # Drop unnecessary columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def get_multi_week_intraday_data(self, symbol, weeks=2, interval="1min"):
        """
        Get multiple weeks of intraday data by combining multiple API calls if needed.
        """
        print(f"Retrieving {weeks} weeks of {interval} data for {symbol}...")
        
        if self.api_provider == "polygon":
            # Calculate dates
            end_date = datetime.now()
            all_data = []
            
            # Polygon allows 2 years of minute data, but we'll break it into 
            # smaller chunks to avoid hitting limitations
            chunk_size = 30  # days per request
            total_days = weeks * 7
            
            print(f"Total days requested: {total_days}")
            print(f"Current end date: {end_date.date()}")
            
            for i in range(0, total_days, chunk_size):
                chunk_end = end_date - timedelta(days=i)
                chunk_start = chunk_end - timedelta(days=min(chunk_size, total_days - i))
                
                print(f"\nFetching data chunk from {chunk_start.date()} to {chunk_end.date()}")
                
                # Format dates
                start_str = chunk_start.strftime('%Y-%m-%d')
                end_str = chunk_end.strftime('%Y-%m-%d')
                
                # Adjust interval format for Polygon
                timespan = interval.replace("min", "")
                
                # Construct URL
                url = f"{self.base_url}/v2/aggs/ticker/{symbol.upper()}/range/{timespan}/minute/{start_str}/{end_str}"
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                self._throttle_api_calls()
                response = requests.get(url, headers=headers)
                
                if response.status_code != 200:
                    print(f"Warning: API request failed with status code {response.status_code}: {response.text}")
                    continue
                
                data = response.json()
                
                # Add more detailed error checking
                if data.get('status') == 'ERROR':
                    print(f"API returned an error: {data.get('error')}")
                    continue
                    
                results = data.get('results', [])
                
                if results:
                    df_chunk = pd.DataFrame(results)
                    
                    # Print date range of this chunk
                    if 't' in df_chunk.columns:
                        min_date = pd.to_datetime(df_chunk['t'].min(), unit='ms').date()
                        max_date = pd.to_datetime(df_chunk['t'].max(), unit='ms').date()
                        print(f"Chunk data range: {min_date} to {max_date}")
                    
                    # Rename columns
                    df_chunk.rename(columns={
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                        "t": "timestamp"
                    }, inplace=True)
                    
                    # Convert timestamp (milliseconds) to datetime
                    df_chunk['datetime'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                    
                    # Select relevant columns
                    df_chunk = df_chunk[['datetime', 'open', 'high', 'low', 'close', 'volume']]
                    
                    all_data.append(df_chunk)
                    print(f"Retrieved {len(df_chunk)} data points")
                else:
                    print("No data found for this date range")
            
            if not all_data:
                raise Exception("No data retrieved for any date range")
                
            # Combine all chunks
            df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset='datetime')
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Print final date range
            print(f"\nFinal data range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
            print(f"Total data points: {len(df)}")
            
            return df
            
        else:
            # Alpha Vantage approach
            # First try to get as much data as possible in one call
            df = self.get_intraday_stock_data(symbol, interval=interval, outputsize="full")
            
            # Check if we got enough data
            earliest_date = df['datetime'].min().date()
            latest_date = df['datetime'].max().date()
            requested_earliest = datetime.now().date() - timedelta(weeks=weeks)
            
            print(f"Retrieved data from {earliest_date} to {latest_date}")
            
            if earliest_date <= requested_earliest:
                print(f"Successfully retrieved all requested data in one API call")
                
                # Filter to the requested time frame
                mask = df['datetime'].dt.date >= requested_earliest
                return df[mask].sort_values('datetime')
            else:
                print(f"Warning: Could only retrieve data back to {earliest_date}.")
                print(f"This is less than the {weeks} weeks requested (back to {requested_earliest}).")
                print(f"Many free APIs have limited historical intraday data available.")
                
                # Return what we have, sorted by datetime
                return df.sort_values('datetime')
    
    def save_data(self, df, filename=None):
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            filename (str, optional): Name of the file. If None, a default name will be generated.
            
        Returns:
            str: Path to the saved CSV file
        """
        if filename is None:
            # Generate a default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}"
        else:
            # Remove any existing extensions
            filename = filename.replace('.csv', '')
        
        # Ensure the directory exists
        os.makedirs("data/output", exist_ok=True)
        
        # Save to CSV
        csv_filepath = f"data/output/{filename}.csv"
        df.to_csv(csv_filepath, index=False)
        
        return csv_filepath
    

# Example usage
if __name__ == "__main__":
    # Use the Polygon API key
    api_key = "Exp1ROYMF_eyvW6fXJ8N9j89tNqpsAsH"
    
    # Create retriever with provided key for Polygon.io
    retriever = StockDataRetriever(api_key=api_key, api_provider="polygon")
    
    # Save the API key for future use
    retriever.save_api_key(api_key)
    print("Polygon API key saved for future use.")
    
    # Example: Get 2 weeks of minute data for Apple
    symbol = input("Enter stock symbol (default: AAPL): ") or "SPY"
    weeks = int(input("Number of weeks of data to retrieve (default: 2): ") or "2")
    interval = input("Data interval in minutes (1, 5, 15, 30, 60) (default: 1): ") or "1"
    
    stock_data = retriever.get_multi_week_intraday_data(symbol, weeks=weeks, interval=f"{interval}")
    
    # Generate filename
    filename = f"{symbol}_{weeks}weeks_{interval}min_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save to CSV
    csv_path = retriever.save_data(stock_data, filename)
    print(f"Data saved to CSV: {csv_path}")
    
    # Show sample
    print("\nSample data:")
    print(stock_data.head())
