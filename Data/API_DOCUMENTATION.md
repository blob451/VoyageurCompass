# VoyageurCompass Data API Documentation

## Overview
The Data API provides comprehensive endpoints for stock market data retrieval, portfolio management, and market analytics powered by Yahoo Finance integration.

## API Host Configuration
**Important:** Replace `{API_HOST}` in all URLs below with your actual API host:
- **Development:** `localhost:8000`
- **Staging:** `staging-api.yourcompany.com`  
- **Production:** `api.yourcompany.com`

### Example URL Replacements
- Template: `http://{API_HOST}/api/data/stocks/`
- Development: `http://localhost:8000/api/data/stocks/`
- Production: `http://api.yourcompany.com/api/data/stocks/`

## Base URL
```
http://{API_HOST}/api/data/
```

## Authentication
- Public endpoints: No authentication required
- Private endpoints: JWT token required in Authorization header
  ```
  Authorization: Bearer <your-jwt-token>
  ```

## API Endpoints

### Stock Endpoints

#### 1. List All Stocks
- **GET** `/api/data/stocks/`
- **Auth**: Not required
- **Query Params**:
  - `search`: Filter by symbol, name, sector, or industry
  - `ordering`: Sort by symbol, market_cap, last_sync
- **Response**: List of stocks with basic information

#### 2. Get Stock Details
- **GET** `/api/data/stocks/{id}/`
- **Auth**: Not required
- **Response**: Detailed stock information including latest price

#### 3. Get Stock Price History
- **GET** `/api/data/stocks/{id}/prices/`
- **Auth**: Not required
- **Query Params**:
  - `days`: Number of days of history (default: 30)
- **Response**: Array of historical prices

#### 4. Get Stock Historical Data
- **GET** `/api/data/stocks/{id}/historical/`
- **Auth**: Not required
- **Query Params**:
  - `start_date`: ISO format date
  - `end_date`: ISO format date
- **Response**: Historical price data within date range

#### 5. Get Company Information
- **GET** `/api/data/stocks/{id}/info/`
- **Auth**: Not required
- **Response**: Detailed company information from Yahoo Finance

#### 6. Sync Stock Data
- **POST** `/api/data/stocks/{id}/sync/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "period": "1mo"  // Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
  }
  ```
- **Response**: Updated stock data

#### 7. Search Stocks
- **GET** `/api/data/stocks/search/`
- **Auth**: Not required
- **Query Params**:
  - `q`: Search query (symbol or name)
- **Response**: List of matching stocks

#### 8. Get Trending Stocks
- **GET** `/api/data/stocks/trending/`
- **Auth**: Not required
- **Response**: List of trending/popular stocks

#### 9. Get Market Indices
- **GET** `/api/data/stocks/indices/`
- **Auth**: Not required
- **Response**: Major market indices (S&P 500, Dow Jones, NASDAQ, etc.)

#### 10. Batch Sync Stocks
- **POST** `/api/data/stocks/batch_sync/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "period": "1mo"
  }
  ```
- **Response**: Sync results for all symbols

#### 11. Get Real-time Quotes
- **POST** `/api/data/stocks/realtime_quotes/`
- **Auth**: Not required
- **Body**:
  ```json
  {
    "symbols": ["AAPL", "MSFT", "GOOGL"]
  }
  ```
- **Response**: Real-time quotes for specified symbols

### Portfolio Endpoints

#### 1. List User Portfolios
- **GET** `/api/data/portfolios/`
- **Auth**: Required
- **Response**: List of user's portfolios

#### 2. Create Portfolio
- **POST** `/api/data/portfolios/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "name": "My Portfolio",
    "description": "Long-term investments",
    "initial_value": 10000.00,
    "risk_tolerance": "moderate"  // Options: conservative, moderate, aggressive
  }
  ```

#### 3. Get Portfolio Details
- **GET** `/api/data/portfolios/{id}/`
- **Auth**: Required
- **Response**: Portfolio details with holdings

#### 4. Update Portfolio
- **PATCH** `/api/data/portfolios/{id}/`
- **Auth**: Required
- **Body**: Portfolio fields to update

#### 5. Delete Portfolio
- **DELETE** `/api/data/portfolios/{id}/`
- **Auth**: Required

#### 6. Add Holding to Portfolio
- **POST** `/api/data/portfolios/{id}/add_holding/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "stock_symbol": "AAPL",
    "quantity": 10,
    "average_price": 150.00,
    "purchase_date": "2024-01-15"
  }
  ```

#### 7. Update Portfolio Holding
- **POST** `/api/data/portfolios/{id}/update_holding/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "symbol": "AAPL",
    "quantity": 15,
    "average_price": 155.00
  }
  ```

#### 8. Remove Portfolio Holding
- **POST** `/api/data/portfolios/{id}/remove_holding/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "symbol": "AAPL"
  }
  ```

#### 9. Get Portfolio Performance
- **GET** `/api/data/portfolios/{id}/performance/`
- **Auth**: Required
- **Response**: Portfolio performance metrics including gains/losses

#### 10. Update Portfolio Prices
- **POST** `/api/data/portfolios/{id}/update_prices/`
- **Auth**: Required
- **Response**: Updated holdings with latest prices

#### 11. Get Portfolio Allocation
- **GET** `/api/data/portfolios/{id}/allocation/`
- **Auth**: Required
- **Response**: Portfolio allocation by stock, sector, and industry

### Market Data Endpoints

#### 1. Market Overview
- **GET** `/api/data/market/overview/`
- **Auth**: Not required
- **Response**: Market status, indices, top gainers/losers
- **Cache**: 5 minutes

#### 2. Sector Performance
- **GET** `/api/data/market/sectors/`
- **Auth**: Not required
- **Response**: Performance metrics grouped by sector
- **Cache**: 10 minutes

#### 3. Economic Calendar
- **GET** `/api/data/market/calendar/`
- **Auth**: Not required
- **Query Params**:
  - `days`: Number of days ahead (default: 7)
- **Response**: Upcoming economic events and earnings

#### 4. Compare Stocks
- **POST** `/api/data/compare/`
- **Auth**: Not required
- **Body**:
  ```json
  {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "metrics": ["price", "change_percent", "volume", "market_cap"]
  }
  ```
- **Response**: Side-by-side comparison of stocks

### Synchronization Endpoints

#### 1. Sync Watchlist
- **POST** `/api/data/sync/watchlist/`
- **Auth**: Required
- **Body**:
  ```json
  {
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "period": "1mo"
  }
  ```
- **Response**: Synchronization results

#### 2. Bulk Price Update
- **POST** `/api/data/sync/bulk-update/`
- **Auth**: Required
- **Response**: Update status for all active stocks

### Price Data Endpoints

#### 1. List Stock Prices
- **GET** `/api/data/prices/`
- **Auth**: Not required
- **Query Params**:
  - `symbol`: Filter by stock symbol
  - `start_date`: Filter by start date
  - `end_date`: Filter by end date
  - `ordering`: Sort by date, close, volume

### Holdings Endpoints

#### 1. List User Holdings
- **GET** `/api/data/holdings/`
- **Auth**: Required
- **Response**: All holdings across user's portfolios

#### 2. Create Holding
- **POST** `/api/data/holdings/`
- **Auth**: Required
- **Body**: Holding details

#### 3. Update Holding
- **PATCH** `/api/data/holdings/{id}/`
- **Auth**: Required
- **Body**: Fields to update

#### 4. Delete Holding
- **DELETE** `/api/data/holdings/{id}/`
- **Auth**: Required

## Response Formats

### Success Response
```json
{
  "data": {...},
  "message": "Success message",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response
```json
{
  "error": "Error message",
  "details": {...},
  "status": 400
}
```

## Rate Limiting
- Public endpoints: 100 requests per minute
- Authenticated endpoints: 1000 requests per minute
- Bulk operations: 10 requests per minute

## Data Sources
- Primary: Yahoo Finance API (via yfinance)
- Cache: Redis (5-60 minute TTL)
- Database: PostgreSQL

## WebSocket Support (Future)
Real-time price updates will be available via WebSocket connections at:
```
ws://{API_HOST}/ws/stocks/
```

## Examples

### Get Stock with Price History
```bash
curl -X GET "http://{API_HOST}/api/data/stocks/1/prices/?days=7"
```

### Create Portfolio and Add Holdings
```bash
# Create portfolio
curl -X POST "http://{API_HOST}/api/data/portfolios/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Tech Portfolio", "description": "Technology stocks"}'

# Add holding
curl -X POST "http://{API_HOST}/api/data/portfolios/1/add_holding/" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"stock_symbol": "AAPL", "quantity": 10, "average_price": 150.00}'
```

### Compare Multiple Stocks
```bash
curl -X POST "http://{API_HOST}/api/data/compare/" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "metrics": ["price", "market_cap"]}'
```

## Notes
- All decimal values use 2-4 decimal places for precision
- Dates should be in ISO 8601 format
- Stock symbols should be uppercase
- Market data is delayed by 15-20 minutes for free tier
- Historical data limited to 2 years for most endpoints