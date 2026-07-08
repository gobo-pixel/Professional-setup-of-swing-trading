# Quant Trading Platform Architecture

## Overview
This project follows a modular institutional trading architecture.

## High Level Architecture

Orchestrator
→ Watchlist
→ Data Engine
→ Feature Engineering
→ Market Regime
→ Strategy
→ Decision
→ Risk
→ Position Sizing
→ Portfolio Rules
→ Market Scanner
→ Broker
→ Tracker
→ Portfolio
→ Analytics
→ Output

## Core Principles

- Single Responsibility Principle
- Modular Architecture
- Dependency Injection
- No Circular Imports
- Deterministic Backtesting
- Explicit API Contracts

## Source of Truth

The project API is defined by IDL.md.
