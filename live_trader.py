import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import ccxt
import math
import aiohttp
import os
from core.feature_engine import build_advanced_dataset
from core.agent_team import ChiefTraderAgent
from dotenv import load_dotenv

load_dotenv()

# 从环境变量读取 API Key
API_KEY = os.getenv("BINANCE_API_KEY", "")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

if not API_KEY or not SECRET_KEY:
    print("❌ 错误: 未检测到 API Key，请检查根目录的 .env 文件是否正确配置。")
    exit()

exchange_ts = ccxt.binanceusdm({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})
exchange_ts.enable_demo_trading(True)

SYMBOL = "BTC/USDT:USDT" # 注意：如果你想测其他币种，把 btcusdt 改成 actusdt 等
WS_URL = "wss://fstream.binancefuture.com/stream"

# 全局状态

data_queue = asyncio.Queue()
INITIAL_DAILY_BALANCE = None # 记录当天的初始余额
MAX_DAILY_DRAWDOWN = 0.05    # 最大单日回撤 5%
history_klines = []
latest_orderbook = {'bids': [], 'asks': []}
latest_obi = 0.5
latest_news = "暂无新闻"
latest_whale_alert = "暂无巨鲸异动"
latest_fng = 50
agent_team = ChiefTraderAgent(model_name = "finance-expert")

print("正在加载 LightGBM 模型...")
model = lgb.Booster(model_file = "lgb_trading_model.txt")

def update_state_file(**kwargs):
    """更新监控大屏的状态文件"""
    state = {}
    if os.path.exists("live_state.json"):
        try:
            with open("live_state.json", "r", encoding="utf-8") as f:
                state = json.load(f)
        except: pass
    state.update(kwargs)
    with open("live_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

async def place_conditional_order(side, amount, trigger_price, order_type):
    """底层原生 API 挂条件单"""
    raw_symbol = SYMBOL.split(':')[0].replace('/', '')
    params = {
        'algoType': 'CONDITIONAL',       # 👈 必须声明是条件单
        'symbol': raw_symbol,
        'side': side.upper(),            # 'BUY' 或 'SELL'
        'type': order_type.upper(),      # 'STOP_MARKET' 或 'TAKE_PROFIT_MARKET'
        'quantity': amount,
        'triggerPrice': trigger_price,   # 触发价
        'reduceOnly': 'true',            # 只减仓
        'workingType': 'MARK_PRICE',     # 标记价格触发
        'timeInForce': 'GTC'             # 市价条件单必须带 GTC
    }
    return await asyncio.to_thread(exchange_ts.fapiPrivatePostAlgoOrder, params)

async def cancel_all_orders():
    """一键撤销该交易对的所有活动委托（防止脏单堆积）"""
    try:
        await asyncio.to_thread(exchange_ts.cancel_all_orders, SYMBOL)
        print("🗑️ 已清理所有旧委托单。")
    except Exception as e:
        pass # 如果没有单子，会报错，忽略即可

async def trailing_stop_worker():
    """每 5 秒检查一次，如果浮盈超 1%，立刻移动保本止损"""
    print("🛡️ 保本止损巡检器已启动...")
    while True:
        try:
            positions = await asyncio.to_thread(exchange_ts.fetch_positions, [SYMBOL])

            if not positions or float(positions[0]['contracts']) == 0:
                # 检查是否还有残留委托
                open_orders = await asyncio.to_thread(exchange_ts.fetch_open_orders, SYMBOL)
                if open_orders:
                    print("🧹 发现空仓但存在残留委托，正在清扫孤儿单...")
                    await asyncio.to_thread(exchange_ts.cancel_all_orders, SYMBOL)
                    print("✅ 孤儿单已全部清理。")
            
            elif positions and float(positions[0]['contracts']) != 0:
                pos = positions[0]
                entry_price = float(pos['entryPrice'])
                # 计算投资回报率
                pnl_pct = float(pos['unrealizedPnl']) / float(pos['initialMargin'])

                if pnl_pct >= 0.01: # 如果盈利超过 1%
                    open_orders = await asyncio.to_thread(exchange_ts.fetch_open_orders, SYMBOL)
                    for o in open_orders:
                        if o['type'] == 'STOP_MARKET':
                            old_sl = float(o['stopPrice'] if 'stopPrice' in o else o['info']['stopPrice'])
                            if pos['side'] == 'long':
                                target_sl = round(entry_price * 1.001, 2) # 保本 + 覆盖手续费
                                if target_sl > old_sl:
                                    await asyncio.to_thread(exchange_ts.cancel_order, o['id'], SYMBOL)
                                    await asyncio.sleep(1.0)
                                    await place_conditional_order('sell', abs(float(pos['contracts'])), target_sl, 'STOP_MARKET')
                                    print(f"🛡️ [高频巡检] 多单保本止损已上移至: {target_sl}")

                            elif pos['side'] == 'short':
                                target_sl = round(entry_price * 0.999, 2)
                                if target_sl < old_sl:
                                    await asyncio.to_thread(exchange_ts.cancel_order, o['id'], SYMBOL)
                                    await asyncio.sleep(1.0) 
                                    await place_conditional_order('buy', abs(float(pos['contracts'])), target_sl, 'STOP_MARKET')
                                    print(f"🛡️ [高频巡检] 空单保本止损已下移至: {target_sl}")
        except Exception as e:
            pass
        await asyncio.sleep(5) # 每5秒巡检一次

async def websocket_producer():
    """生产者：只负责收数据"""
    # 使用基础合流端点（支持动态订阅）
    print(f"启动 WebSocket 监听：{WS_URL}")
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval = 20, ping_timeout = 20) as ws:
                print("✅ WebSocket 连接成功！")
                # 👈 核心：主动发送订阅指令
                sub_msg = {"method": "SUBSCRIBE", "params": ["btcusdt@kline_1h", "btcusdt@depth5@100ms"], "id": 1}
                await ws.send(json.dumps(sub_msg))
                print("📡 订阅指令已发送，等待行情推送...")
                
                while True:
                    try:
                        # 10 秒超时防假死
                        response = await asyncio.wait_for(ws.recv(), timeout = 10.0)
                        data = json.loads(response)

                        # 👈 核心：只管扔进队列，绝不做任何计算！
                        await data_queue.put(data)
                    except asyncio.TimeoutError:
                        print("⚠️ 10 秒未收到数据，主动重连...")
                        break
        except Exception as e:
            print(f"❌ 连接失败或异常: {e}")
        await asyncio.sleep(5)

async def strategy_consumer():
    """消费者：负责算数据和下单 """
    global latest_obi
    print("策略引擎已启动，等待队列数据...")
    while True:
        # 👈 从队列拿出数据，如果没有数据就会在这里异步等待
        data = await data_queue.get()

        if 'stream' in data and 'data' in data:
            # 1. 如果是深度数据 (订单簿)
            if 'depth' in data['stream']:
                depth = data['data']
                bids, asks = depth['b'], depth['a']
                
                if bids and asks:
                    # 买一价和卖一价
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    # 5档总买量和总卖量
                    total_bid = sum(float(b[1]) for b in bids)
                    total_ask = sum(float(a[1]) for a in asks)
                    
                    if total_bid + total_ask > 0:
                        latest_obi = total_bid / (total_bid + total_ask)
                        best_bid = float(bids[0][0])
                        update_state_file(price = best_bid, obi = latest_obi)
                        print(f"📊 买一: {best_bid} | OBI: {latest_obi * 100:.2f}%", end = "\r")

            elif 'kline' in data['stream']:
                kline_data = data['data']
                k = kline_data['k']
                if not k['x']:
                    print(f"💓 当前行情: {k['c']} | OBI: {latest_obi * 100:.2f}%", end = "\r")
                else:
                    print(f"\n📥 收到 1h K线收线！")
                    await execute_strategy(k)
        data_queue.task_done()

async def execute_strategy(k):
    """下单逻辑"""
    global INITIAL_DAILY_BALANCE
    global latest_obi
    try:
    # 获取真实资金费率
        rate_info = await asyncio.to_thread(exchange_ts.fetch_funding_rate, SYMBOL)
        real_funding_rate = float(rate_info['fundingRate']) if rate_info else 0.0001
    except:
        real_funding_rate = 0.0001

    new_row = {
        'datetime': pd.to_datetime(int(k['t']), unit = 'ms').tz_localize('UTC').tz_convert('Asia/Shanghai'),
        'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']),
        'close': float(k['c']), 'volume': float(k['v']), 'funding_rate': real_funding_rate # 调接口查一下真实的
    }
    history_klines.append(new_row)

    if len(history_klines) > 300: history_klines.pop(0)
    if len(history_klines) < 200: return

    df = pd.DataFrame(history_klines).set_index('datetime')

    # 引入长线趋势过滤（计算 SMA_200）
    sma_200 = df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else df['close'].mean()
    current_price = float(k['c'])
    is_bull_market = current_price > sma_200

    try:
        df_features = build_advanced_dataset(df, is_live = True)
        if df_features.empty: return

        features_list = [
            'MA5', 'MA20', 'EMA10', 'Price_MA20_Ratio', 'RSI', 'MACD', 'MACD_Signal', 
            'MACD_Hist', 'ATR', 'ATR_Ratio', 'BB_Std', 'BB_Width', 'Vol_Change', 
            'Ret_3h', 'Ret_6h', 'ai_sentiment', 'funding_rate',
            
            'Price_SMA200_Ratio', 'RSI_24', 'Vol_Ratio_24', 'Upper_Shadow', 'Lower_Shadow'
        ]
        pred_return = model.predict(df_features.iloc[-1:][features_list])[0]
        print(f"⏰ {new_row['datetime']} | 价格: {current_price} | 预测收益: {pred_return * 100:.4f}% | 牛市: {is_bull_market}")

        # 获取持仓与余额
        positions = await asyncio.to_thread(exchange_ts.fetch_positions, [SYMBOL])
        current_position_amt = float(positions[0]['contracts']) if positions else 0

        #趋势反转平仓逻辑
        if current_position_amt > 0 and pred_return < -0.0005:
            print("⚠️ 趋势反转，主动平多单！")
            try:
                await asyncio.to_thread(exchange_ts.create_market_sell_order, SYMBOL, abs(current_position_amt), {'reduceOnly': True})
            except Exception as e:
                if "-2022" not in str(e):
                    print(f"平多仓时发生其他错误: {e}")
            await cancel_all_orders()
            return
            
        elif current_position_amt < 0 and pred_return > 0.0005:
            print("⚠️ 趋势反转，主动平空单！")
            try:
                await asyncio.to_thread(exchange_ts.create_market_buy_order, SYMBOL, abs(current_position_amt), {'reduceOnly': True})
            except Exception as e:
                if "-2022" not in str(e):
                    print(f"平空仓时发生其他错误: {e}")
            await cancel_all_orders()
            return

        # 全局熔断
        balance_info = await asyncio.to_thread(exchange_ts.fetch_balance)
        current_balance = float(balance_info['USDT']['total'])
        if INITIAL_DAILY_BALANCE is None:
            INITIAL_DAILY_BALANCE = current_balance
        drawdown = (INITIAL_DAILY_BALANCE - current_balance) / INITIAL_DAILY_BALANCE
        if drawdown >= MAX_DAILY_DRAWDOWN:
            print(f"🚨 触发熔断！今日回撤 {drawdown*100:.2f}%")
            return

        # ATR 动态止损
        latest_atr_ratio = df_features['ATR_Ratio'].iloc[-1]
        dynamic_sl_pct = np.clip(latest_atr_ratio * 1.5, 0.01, 0.03)

        # ==========================================
        # 👈 核心改动：呼叫 AI 首席交易员审核
        # ==========================================
        print("🧠 正在呼叫 AI 首席交易员进行最终裁决...")
        ai_decision, ai_reason = await agent_team.make_decision(
            tech_pred_return = pred_return,
            obi = latest_obi,
            is_bull_market = is_bull_market,
            fng_value = latest_fng,
            news_headline = latest_news,
            whale_alert = latest_whale_alert
        )
        print(f"🤵 [首席交易员裁决]: {ai_decision} | 理由: {ai_reason}")
        update_state_file(
            pred = float(pred_return * 100),
            ai_decision = ai_decision,
            ai_reason = ai_reason
        )

        # 如果 AI 觉得风险太大，直接一票否决，结束本次策略
        if ai_decision == 0:
            print("🛑 AI 否决了本次交易，放弃。")
            return
        # ==========================================

        # 凯利公式计算仓位
        # 假设模型预测的收益率绝对值越大，胜率越高
        # 用 Sigmoid 将预测收益映射到 0~1 的胜率（缩放系数设为 50，防止太敏感）
        # 预测收益为 0 时，胜率为 0.5
        win_rate = 1 / (1 + math.exp(-abs(pred_return) * 50))
        b = 2.0
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        safe_fraction = max(0, min(kelly_fraction * 0.5, 0.10))
        account_balance = INITIAL_DAILY_BALANCE if INITIAL_DAILY_BALANCE else 1000
        dynamic_position_size = round(account_balance * safe_fraction, 2)
        # 如果资金太小，直接不交易，防止报错
        if dynamic_position_size < 20:
            print(f"🧮 凯利计算仓位 {dynamic_position_size} U 太小，放弃交易")
            return

        #大趋势过滤 + 开仓
        if ai_decision == 1 and current_position_amt == 0 and is_bull_market:
            print(f"🟢 AI 批准顺势做多！投入: {dynamic_position_size} U")
            await cancel_all_orders() # 开仓前清空脏单
            amount = round(dynamic_position_size * 3 / current_price, 3)
            if amount <= 0: return
            await asyncio.to_thread(exchange_ts.create_market_buy_order, SYMBOL, amount)
            await asyncio.sleep(1.5) 
            
            sl_price = round(current_price * (1 - dynamic_sl_pct), 2)
            tp_price = round(current_price * (1 + 0.03), 2) # 价格涨3%
            await place_conditional_order('sell', amount, sl_price, 'STOP_MARKET')
            await place_conditional_order('sell', amount, tp_price, 'TAKE_PROFIT_MARKET')
            print(f"✅ 开多成功，止损 {sl_price} / 止盈 {tp_price}")

        elif ai_decision == -1 and current_position_amt == 0 and not is_bull_market:
            print(f"🔴 AI 批准顺势做空！投入: {dynamic_position_size} U")
            await cancel_all_orders() # 开仓前清空脏单
            amount = round(dynamic_position_size * 3 / current_price, 3)
            if amount <= 0: return
            await asyncio.to_thread(exchange_ts.create_market_sell_order, SYMBOL, amount)
            await asyncio.sleep(1.5) 
            
            sl_price = round(current_price * (1 + dynamic_sl_pct), 2)
            tp_price = round(current_price * (1 - 0.03), 2)
            await place_conditional_order('buy', amount, sl_price, 'STOP_MARKET')
            await place_conditional_order('buy', amount, tp_price, 'TAKE_PROFIT_MARKET')
            print(f"✅ 开空成功，止损 {sl_price} / 止盈 {tp_price}")

    except Exception as e:
        print(f"❌ 策略执行出错: {e}")

async def fetch_news_background():
    """后台任务：并发抓取新闻"""
    global latest_news, latest_fng
    print("📰 新闻并发抓取后台已启动...")
    while True:
        try:
            async with aiohttp.ClientSession(timeout = aiohttp.ClientTimeout(total = 5)) as session:
                task1 = session.get("https://api.alternative.me/fng/")
                task2 = session.get("https://cryptocurrency.cv/api/news?limit=3")
                responses = await asyncio.gather(task1, task2, return_exceptions = True)
                for i, res in enumerate(responses):
                    if isinstance(res, aiohttp.ClientResponse) and res.status == 200:
                        data = await res.json()
                        if i == 0:
                            latest_fng = int(data['data'][0]['value'])
                            update_state_file(fng = latest_fng)
                        elif i == 1:
                            if isinstance(data, list) and len(data) > 0:
                                latest_news = data[0]['title']
                                update_state_file(news = latest_news)
        except Exception:
            pass
        await asyncio.sleep(300) # 每 5 分钟抓取一次

async def onchain_worker():
    """后台任务：监控链上巨鲸转账 (基于 mempool.space 免费接口)"""
    global latest_whale_alert
    print("🐋 链上巨鲸监控后台已启动...")
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                # 获取最新的未确认交易
                async with session.get("https://mempool.space/api/mempool/recent", timeout = 10) as response:
                    if response.status == 200:
                        txs = await response.json()
                        for tx in txs:
                            value_btc = tx['value'] / 100_000_000
                            if value_btc >= 10:
                                latest_whale_alert = f"检测到一笔 {value_btc:.2f} BTC 的大额转账正在打包，请注意市场抛压/承接！"
                                print(f"🚨 [链上警报] {latest_whale_alert}")
                                break # 抓到一个大的就记录，避免刷屏
                        else:
                            latest_whale_alert = "链上暂无大额异动"
        except Exception as e:
            pass
        await asyncio.sleep(120) # 每 2 分钟抓取一次

async def main():
    print("🔥 正在预热 AI 首席交易员模型...")
    try:
        # 随便发一个测试，强制 Ollama 把模型加载进显存
        await agent_team.make_decision(
            tech_pred_return = 0.0, obi = 0.5, is_bull_market = True, fng_value = 50, news_headline="预热测试"
        )
        print("✅ AI 模型预热完毕！")
    except Exception as e:
        print(f"⚠️ 预热失败: {e}")

    print("正在拉取历史 K 线初始化缓存...")

    exchange_init = ccxt.binanceusdm()
    initial = exchange_init.fetch_ohlcv(SYMBOL, '1h', limit = 300)
    for o in initial:
        history_klines.append({
            'datetime': pd.to_datetime(int(o[0]), unit = 'ms').tz_localize('UTC').tz_convert('Asia/Shanghai'),
            'open': float(o[1]), 'high': float(o[2]), 'low': float(o[3]), 
            'close': float(o[4]), 'volume': float(o[5]), 'funding_rate': 0.0001 
        })
    print(f"✅ 历史数据初始化完成，当前缓存: {len(history_klines)} 根")

    # 👈 核心新增：强行手动测试一次！
    print("\n--- ⚡ 开始执行一次性假数据测试 ---")
    latest_price = history_klines[-1]['close']
    fake_k = {
        't': int(time.time() * 1000),
        'o': str(latest_price), 
        'h': str(latest_price), 
        'l': str(latest_price), 
        'c': str(latest_price), 
        'v': '100',
        'x': True 
    }
    # 直接调用策略引擎，不用经过队列
    await execute_strategy(fake_k) 
    print("--- ⚡ 假数据测试结束，正式进入监听 ---\n")

    print("🚀 启动高频交易引擎...")
    await asyncio.gather(
        websocket_producer(),
        strategy_consumer(),
        fetch_news_background(),
        trailing_stop_worker(),
        onchain_worker()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户手动停止程序。")
