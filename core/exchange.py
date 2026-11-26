import ccxt
import pandas as pd
import time
import requests

class BinanceDataFetcher:
    def __init__(self, is_testnet = True):
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        if is_testnet:
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 3000): 
        """
        获取历史K线数据，并分页拉取指定数量的历史 K 线（突破单次 1000 根的限制）
        :param symbol: 交易对，例如 'BTC/USDT'
        :param timeframe: 时间周期，例如 '1m', '5m', '1h', '4h', '1d' （可改）
        :param limit: 获取数量（最多 1500 条）
        """

        print(f"正在从币安分页拉取 {symbol} 的 {timeframe} K线，目标 {limit} 根...")

        all_ohlcv = []
        now = self.exchange.milliseconds()
        timeframe_ms = self.exchange.parse_timeframe(timeframe) * 1000
        batch_limit = 1000
        since = now - limit * timeframe_ms

        while len(all_ohlcv) < limit:
            try:
                # 每次拉 1000 根（接口限制）
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since = since, limit = batch_limit)
                if not ohlcv: break

                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + timeframe_ms

                if since >= now: break
                time.sleep(0.5)
            except Exception as e:
                print(f"分页拉取中断: {e}")
                break

        all_ohlcv = list({v[0]: v for v in all_ohlcv}.values())
        all_ohlcv.sort(key = lambda x: x[0])
        all_ohlcv = all_ohlcv[-limit:]

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        print(f"拉取完成，实际获取 {len(df)} 根 K 线。")
        return df

    def fetch_funding_rate(self, symbol: str):
        """直接请求币安底层 API 获取资金费率"""
        try:
            raw_symbol = symbol.split(':')[0].replace('/', '')
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={raw_symbol}"
            response = requests.get(url, timeout = 5)
            data = response.json()
            if 'lastFundingRate' in data:
                return float(data['lastFundingRate'])
            return None
        except Exception as e:
            print(f"获取资金费率失败: {e}")
            return None

    def fetch_funding_rate_history(self, symbol: str, limit: int = 1000):
        """拉取历史资金费率"""
        try:
            # 将 'BTC/USDT:USDT' 转成 'BTCUSDT'
            raw_symbol = symbol.split(':')[0].replace('/', '')
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={raw_symbol}&limit={limit}"
            res = requests.get(url, timeout=10).json()
            
            df_rate = pd.DataFrame(res)
            df_rate['datetime'] = pd.to_datetime(df_rate['fundingTime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
            df_rate['funding_rate'] = df_rate['fundingRate'].astype(float)
            return df_rate[['datetime', 'funding_rate']].set_index('datetime')
        except Exception as e:
            print(f"获取资金费率历史失败: {e}")
            return None


if __name__ == "__main__":
    fetcher = BinanceDataFetcher(is_testnet = False)
    symbol = "BTC/USDT:USDT" #可改
    df = fetcher.fetch_ohlcv(symbol, "1h", 100)

    if df is not None:
        print(df.tail()) # 打印最后5根K线
        print(f"\n当前 {symbol} 资金费率: {fetcher.fetch_funding_rate(symbol)}")
