# Quant SDK Quick Reference

## Task Order
1. data_collection
2. feature_engineering
3. signal_generation
4. risk_management
5. backtesting

## Preferred Modules
- quant_data.stock_collector.price_collector.collector
- quant_langchain.features.momentum
- quant_langchain.signals.rule_engine
- quant_langchain.risk.position_manager
- quant_langchain.backtest.engine

## Risk Defaults
- max_position_size: 0.1
- stop_loss: 0.02
- max_drawdown: 0.2

## Backtest Defaults
- initial_cash: 100000
- fee_bps: 5
- window: 2y
- rebalance: daily
