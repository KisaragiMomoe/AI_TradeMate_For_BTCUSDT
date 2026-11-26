import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, initial_balance = 1000.00, fee_rate = 0.0004, leverage = 1.0, stop_loss_pct = 0.04, cooldown_hours = 12):
        """
        :param initial_balance: 初始资金 (USDT)
        :param fee_rate: 手续费率 (币安U本位合约Taker默认是 0.0004，即万分之四)
        :param leverage: 杠杆倍数
        :param stop_loss_pct: 止损百分比
        :param cooldown_hours: 平仓之后冷静时间 
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.stop_loss_pct = stop_loss_pct
        self.cooldown_hours = cooldown_hours
        
        self.last_close_time = None
        self.position = 0
        self.entry_price = 0.0
        self.qty = 0.0
        self.trades = []
        self.equity_curve = []

    def run(self, df, strategy_func):
        """
        运行回测
        :param df: 包含技术指标的 K 线数据
        :param strategy_func: 交易策略函数 (输入行情数据，输出 1, -1, 或 0)
        """
        print("开始回测...")

        for i in range(20, len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]

            if self.last_close_time is not None:
                time_diff = (current_time - self.last_close_time).total_seconds() / 3600
                if time_diff < self.cooldown_hours:
                    self.equity_curve.append({"time": current_time, "balance": self.balance})
                    continue

            #加了个止损 方向别写错
            if self.position != 0:
                if self.position == 1: current_pnl_pct = (self.entry_price - current_price) / self.entry_price
                else: current_pnl_pct = (current_price - self.entry_price) / self.entry_price

                if current_pnl_pct >= self.stop_loss_pct:
                    print(f"🚨 触发止损！当前亏损: {current_pnl_pct*100:.2f}%")
                    #平仓先
                    if self.position == 1: pnl = (current_price - self.entry_price) * self.qty
                    else: pnl = (self.entry_price - current_price) * self.qty
                    self.balance += pnl
                    fee = self.qty * current_price * self.fee_rate
                    self.balance -= fee

                    self.trades.append({"time": current_time, "type": "止损平仓", "price": current_price, "pnl": pnl, "pnl_pct": -self.stop_loss_pct})
                    self.position = 0
                    self.entry_price = 0.0
                    self.qty = 0.0
                    continue 
            #止损结束

            target_signal = strategy_func(df, i)
            if target_signal is None: continue

            if target_signal != self.position:

                # 1. 先平仓
                pnl = 0
                if self.position != 0:
                    if self.position == 1: pnl = (current_price - self.entry_price) * self.qty
                    else: pnl = (self.entry_price - current_price) * self.qty 
                self.balance += pnl

                fee = self.qty * current_price * self.fee_rate
                self.balance -= fee

                margin_used = self.qty * self.entry_price / self.leverage
                pnl_pct = pnl / margin_used if margin_used > 0 else 0

                self.trades.append({"time": current_time, "type": "平仓", "price": current_price, "pnl": pnl, "pnl_pct": pnl_pct})
                self.position = 0
                self.entry_price = 0.0
                self.qty = 0.0

                # 2. 开新仓
                if target_signal == 1:
                    self.position = 1
                    self.entry_price = current_price
                    self.qty = self.balance * self.leverage / current_price
                    fee = self.qty * current_price * self.fee_rate
                    self.balance -= fee
                    self.trades.append({"time": current_time, "type": "开多", "price": current_price})
                
                elif target_signal == -1:
                    self.position = -1
                    self.entry_price = current_price
                    self.qty = self.balance * self.leverage / current_price
                    fee = self.qty * current_price * self.fee_rate
                    self.balance -= fee
                    self.trades.append({"time": current_time, "type": "开空", "price": current_price})

                self.equity_curve.append({"time": current_time, "balance": self.balance})

        if self.position != 0:
            current_price = df['close'].iloc[-1]
            if self.position == 1: pnl = (current_price - self.entry_price) * self.qty
            else: pnl = (self.entry_price - current_price) * self.qty
            self.balance += pnl 
            fee = self.qty * current_price * self.fee_rate
            self.balance -= fee
            self.position = 0
            self.entry_price = 0
            self.qty = 0
            self.equity_curve[-1]["balance"] = self.balance

        self.report()

    def report(self):
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        print("="*40)
        print(f"初始资金: {self.initial_balance:.2f} | 杠杆: {self.leverage}x")
        print(f"最终资金: {self.balance:.2f} USDT")
        print(f"总收益率: {total_return:.2f}%")
        print(f"交易次数: {len(self.trades)//2} 次完整交易")
        
        # 计算胜率
        closed_trades = [t for t in self.trades if t["type"] == "平仓"]
        if closed_trades:
            win_trades = [t for t in closed_trades if t["pnl"] > 0]
            win_rate = len(win_trades) / len(closed_trades) * 100
            print(f"交易胜率: {win_rate:.2f}%")
            
        print("="*40)