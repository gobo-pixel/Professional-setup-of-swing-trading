# Interface Definition Language (IDL)

## Data Layer

MarketDataProvider.fetch(symbol, interval='1d', period='1y') -> DataFrame

FundamentalDataProvider.fetch(symbol) -> dict

NewsDataProvider.fetch(symbol, limit=20) -> list[dict]

DataEngine.fetch(symbol, interval='1d', period='1y', news_limit=20) -> DataBundle

## Feature Layer

TechnicalFeatureEngine.generate(dataframe) -> DataFrame

FeatureEngineeringEngine.generate(dataframe) -> DataFrame

MarketRegimeEngine.evaluate(dataframe) -> DataFrame

## Strategy Layer

BuyStrategyEngine.evaluate(dataframe, fundamentals, news_score, market_score, sector_score) -> BuyDecision

SellStrategyEngine.evaluate(...) -> SellDecision

BuyScoringEngine.calculate(...) -> BuyScore

SellScoringEngine.calculate(...) -> SellScore

BuyProbabilityEngine.calculate(...) -> BuyProbability

SellProbabilityEngine.calculate(...) -> SellProbability

## Decision Layer

DecisionEngine.evaluate(...) -> FinalDecision

ValidationEngine.validate(...) -> ValidationResult

## Risk Layer

RiskManager.evaluate(...) -> RiskResult

PositionSizingEngine.calculate(...) -> PositionResult

PortfolioRulesEngine.evaluate(...) -> PortfolioResult

## Execution Layer

MarketScanner.scan_symbol(...) -> ScanResult

MarketScanner.scan_symbols(...) -> list[ScanResult]

BrokerEngine.place_order(...) -> OrderResult

PositionTracker.update(...) -> list[TrackerResult]

IDL Version: 1.0
