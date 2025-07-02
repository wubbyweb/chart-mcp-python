# Chart MCP Python

A Python implementation of the Model Context Protocol (MCP) server that generates TradingView chart visualizations using the Chart-IMG API. Built with FastMCP for high performance and supports multiple transport protocols.

## Features

- **FastMCP Implementation**: High-performance MCP server using FastMCP
- **Multiple Transports**: STDIO, HTTP, and SSE support
- **Chart Duration Control**: Specify timeframes (1D, 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y)
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Professional Charts**: TradingView-quality visualizations
- **Type Safety**: Full Pydantic model validation

## Quick Start

### 1. Get API Key
Register at [chart-img.com](https://chart-img.com) to get your API key.

### 2. Installation

```bash
# Clone and navigate to Python directory
cd chart-mcp-python

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Set API key
export CHART_IMG_API_KEY="your_api_key_here"
```

### 3. Run MCP Server

**STDIO Mode (Claude Desktop)**
```bash
chart-mcp-server
```

**HTTP Mode (Universal)**
```bash
chart-mcp-http
# Server runs on: http://localhost:3001
```

**SSE Mode (Real-time)**
```bash
chart-mcp-sse
# Server runs on: http://localhost:3002
```

## MCP Client Configuration

### Claude Desktop

Add to `claude_desktop_config.json`:

**STDIO Mode:**
```json
{
  "mcpServers": {
    "chart-server-python": {
      "command": "chart-mcp-server",
      "env": {
        "CHART_IMG_API_KEY": "your_api_key"
      }
    }
  }
}
```

**HTTP Mode:**
```json
{
  "mcpServers": {
    "chart-server-python-http": {
      "url": "http://localhost:3001/mcp",
      "transport": "http"
    }
  }
}
```

**SSE Mode:**
```json
{
  "mcpServers": {
    "chart-server-python-sse": {
      "url": "http://localhost:3002/sse",
      "transport": "sse"
    }
  }
}
```

**Config Locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

## Chart Creation Workflow

⚠️ **IMPORTANT**: This MCP server uses a **three-step workflow** for chart creation:

1. **`initialize_chart`** - Create a chart request with basic parameters
2. **`add_[indicator]`** - Add technical indicators (optional, repeatable)
3. **`download_chart`** - Generate and download the final chart

**You must maintain the same `requestId` across all three steps.**

## Available Tools

### Core Workflow Tools

#### `initialize_chart`
Initialize a new chart request with basic parameters.

**Required Parameters:**
- `symbol` - Trading symbol (e.g., "NASDAQ:AAPL")
- `interval` - Time interval ("1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1D", "1W", "1M")
- `chartType` - Chart type ("candlestick", "line", "area", "bar", "heikin_ashi", "hollow_candle", "baseline", "hi_lo", "column")

**Optional Parameters:**
- `duration` - Chart timeframe ("1D", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y") - default: "1M"
- `width` - Chart width (400-2000) - default: 800
- `height` - Chart height (300-1500) - default: 600
- `theme` - Chart theme ("light", "dark") - default: "light"
- `showVolume` - Show volume indicator - default: true
- `showGrid` - Show chart grid - default: true
- `timezone` - Chart timezone - default: "America/New_York"

**Returns:** A `requestId` that must be used in subsequent calls.

#### `download_chart`
Generate and download the final chart with all configured indicators.

**Required Parameters:**
- `requestId` - The request ID from `initialize_chart`

**Returns:** Chart generation result with processing time and status.

### Technical Indicator Tools

All indicator tools require the `requestId` from `initialize_chart`:

#### Moving Average Indicators
- **`add_sma`** - Simple Moving Average
- **`add_ema`** - Exponential Moving Average
- **`add_wma`** - Weighted Moving Average
- **`add_hull_ma`** - Hull Moving Average

#### Oscillator Indicators
- **`add_rsi`** - Relative Strength Index
- **`add_macd`** - MACD
- **`add_stochastic`** - Stochastic Oscillator
- **`add_cci`** - Commodity Channel Index
- **`add_williams_r`** - Williams %R

#### Volatility Indicators
- **`add_bollinger_bands`** - Bollinger Bands
- **`add_atr`** - Average True Range
- **`add_keltner_channels`** - Keltner Channels

#### Volume Indicators
- **`add_volume`** - Volume
- **`add_obv`** - On Balance Volume
- **`add_cmf`** - Chaikin Money Flow

#### Trend Indicators
- **`add_adx`** - Average Directional Index
- **`add_parabolic_sar`** - Parabolic SAR
- **`add_ichimoku`** - Ichimoku Cloud

### Utility Tools

#### `get_chart_status`
Check generation progress and retrieve completed charts.

#### `get_available_symbols`
List available trading symbols from Chart-IMG API.

#### `get_recent_requests`
View recent chart generation history.

#### `health_check`
Verify API configuration and service status.

## Usage Examples

### Basic Chart Creation
```bash
# 1. Initialize chart
initialize_chart(symbol="NASDAQ:AAPL", interval="1D", chartType="candlestick")
# Returns: requestId = "req_123456"

# 2. Download chart (no indicators)
download_chart(requestId="req_123456")
```

### Chart with Technical Indicators
```bash
# 1. Initialize chart
initialize_chart(symbol="NASDAQ:AAPL", interval="1D", chartType="candlestick", duration="3M")
# Returns: requestId = "req_789012"

# 2. Add indicators (optional, can add multiple)
add_sma(requestId="req_789012", length=20, color="#FF0000")
add_rsi(requestId="req_789012", length=14, color="#0000FF")
add_bollinger_bands(requestId="req_789012", length=20, mult=2.0)

# 3. Download final chart
download_chart(requestId="req_789012")
```

### Advanced Chart Setup
```bash
# 1. Initialize with custom settings
initialize_chart(
    symbol="BINANCE:BTCUSDT",
    interval="4h",
    chartType="candlestick",
    duration="6M",
    width=1200,
    height=800,
    theme="dark"
)
# Returns: requestId = "req_345678"

# 2. Add multiple indicators
add_ema(requestId="req_345678", length=21, color="#FFD700")
add_ema(requestId="req_345678", length=50, color="#32CD32")
add_macd(requestId="req_345678", fast_length=12, slow_length=26, signal_length=9)
add_volume(requestId="req_345678", show_ma=true, ma_length=20)

# 3. Download chart
download_chart(requestId="req_345678")
```

## AI Tool Integration Prompt

**For AI tools using this MCP server, use this prompt to understand the correct workflow:**

```
This Chart MCP server requires a specific three-step workflow for creating charts:

STEP 1: INITIALIZE CHART
- Always call `initialize_chart` first with basic chart parameters
- Save the returned `requestId` for use in all subsequent calls
- Required: symbol, interval, chartType
- Optional: duration, width, height, theme, showVolume, showGrid, timezone

STEP 2: ADD INDICATORS (Optional)
- Use the `requestId` from step 1
- Call indicator tools like `add_sma`, `add_rsi`, `add_macd`, etc.
- Each indicator tool adds one indicator to the chart
- You can call multiple indicator tools to add multiple indicators
- Common indicators:
  * Moving Averages: add_sma, add_ema (requires: length)
  * Oscillators: add_rsi, add_macd (rsi requires: length; macd requires: fast_length, slow_length, signal_length)
  * Volatility: add_bollinger_bands (requires: length, mult)
  * Volume: add_volume (optional: show_ma, ma_length)

STEP 3: DOWNLOAD CHART
- Call `download_chart` with the same `requestId`
- This generates the final chart with all configured indicators
- Returns the chart data and processing information

CRITICAL: Always use the same requestId across all three steps!

Example workflow:
1. result1 = initialize_chart(symbol="NASDAQ:AAPL", interval="1D", chartType="candlestick")
2. extract requestId from result1
3. add_sma(requestId=extracted_id, length=20)
4. add_rsi(requestId=extracted_id, length=14)
5. download_chart(requestId=extracted_id)

```

## MCP Protocol Endpoints

### HTTP MCP Transport
```bash
POST http://localhost:3001/mcp
```
For use with `@modelcontextprotocol/inspector` and other MCP clients that support HTTP transport.

### SSE MCP Transport
```bash
GET http://localhost:3002/sse?session_id=your-session
POST http://localhost:3002/sse?session_id=your-session
```
For use with `@modelcontextprotocol/inspector` and other MCP clients that support SSE transport.

## HTTP API Endpoints

When running in HTTP mode, the following REST endpoints are available:

### Chart Initialization
```bash
POST /initialize_chart
```

### Chart Status
```bash
GET /chart_status/{request_id}
```

### Available Symbols
```bash
GET /symbols
```

### Recent Requests
```bash
GET /recent_requests?limit=10
```

### Health Check
```bash
GET /health
```

## SSE Real-time Events

When running in SSE mode, connect to the event stream:

```bash
# Connect to SSE stream
curl -N "http://localhost:3001/sse?session_id=my-session"
```

**Event Types:**
- `chart_started` - Chart generation initiated
- `chart_processing` - Chart being processed
- `chart_completed` - Chart generation completed
- `chart_data` - Chart image data available
- `chart_error` - Error occurred

## Testing the Python Implementation

### Basic Testing
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src/chart_mcp_python

# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Linting
ruff src/
```

### Manual Testing
```bash
# Test STDIO mode
echo '{"symbol": "NASDAQ:AAPL", "interval": "1D", "chartType": "candlestick"}' | chart-mcp-server

# Test HTTP mode
curl -X POST http://localhost:3001/initialize_chart \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NASDAQ:AAPL",
    "interval": "1D",
    "chartType": "candlestick",
    "duration": "1M"
  }'

# Test health endpoint
curl http://localhost:3001/health
```

## Sample Usage

### Basic Chart Creation (Python API)
```python
from chart_mcp_python.server.chart_service import chart_service
from chart_mcp_python.shared.schema import ChartConfig

# Create basic chart configuration
config = ChartConfig(
    symbol="NASDAQ:AAPL",
    interval="1D",
    chartType="candlestick",
    duration="3M"
)

# Initialize chart (no indicators)
result = await chart_service.initialize_chart(config, "req_123")
print(f"Chart initialized: {result.success}")
```

### Chart with Indicators via MCP Tools
```python
# This is the recommended approach using MCP tools
# 1. Initialize chart
result1 = await initialize_chart_tool({
    "symbol": "NASDAQ:AAPL",
    "interval": "1D", 
    "chartType": "candlestick",
    "duration": "6M",
    "width": 1200,
    "height": 800,
    "theme": "dark"
})

# Extract requestId from result
request_id = extract_request_id(result1)

# 2. Add indicators
await add_sma_tool({"requestId": request_id, "length": 20, "color": "#FF6B6B"})
await add_ema_tool({"requestId": request_id, "length": 50, "color": "#4ECDC4"})
await add_rsi_tool({"requestId": request_id, "length": 14, "color": "#45B7D1"})

# 3. Download final chart
final_result = await download_chart_tool({"requestId": request_id})
```

## Supported Indicators

### Moving Averages
| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `add_sma` | Simple Moving Average | `length` |
| `add_ema` | Exponential Moving Average | `length` |
| `add_wma` | Weighted Moving Average | `length` |
| `add_hull_ma` | Hull Moving Average | `length` |

### Oscillators
| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `add_rsi` | Relative Strength Index | `length` |
| `add_macd` | MACD | `fast_length`, `slow_length`, `signal_length` |
| `add_stochastic` | Stochastic Oscillator | `k_length`, `k_smoothing`, `d_length` |
| `add_cci` | Commodity Channel Index | `length` |
| `add_williams_r` | Williams %R | `length` |

### Volatility Indicators
| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `add_bollinger_bands` | Bollinger Bands | `length`, `mult` |
| `add_atr` | Average True Range | `length` |
| `add_keltner_channels` | Keltner Channels | `length`, `mult`, `atr_length` |

### Volume Indicators
| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `add_volume` | Volume | None (optional: `show_ma`, `ma_length`) |
| `add_obv` | On Balance Volume | None |
| `add_cmf` | Chaikin Money Flow | `length` |

### Trend Indicators
| Tool Name | Description | Required Parameters |
|-----------|-------------|-------------------|
| `add_adx` | Average Directional Index | `smoothing`, `di_length` |
| `add_parabolic_sar` | Parabolic SAR | `start`, `increment`, `maximum` |
| `add_ichimoku` | Ichimoku Cloud | `conversion_line_length`, `base_line_length`, `leading_span_b_length`, `displacement` |

**All indicator tools also accept optional parameters:**
- `requestId` (required) - The request ID from `initialize_chart`
- `color` (optional) - Hex color code for the indicator
- `linewidth` (optional) - Line width (1-10)
- `plottype` (optional) - Plot type (line, histogram, etc.)

## Symbol Format

Use TradingView symbol format: `EXCHANGE:SYMBOL`

**Examples:**
- `NASDAQ:AAPL` - Apple Inc.
- `NYSE:TSLA` - Tesla Inc.
- `BINANCE:BTCUSDT` - Bitcoin/USDT
- `FOREX:EURUSD` - EUR/USD pair

## Error Handling

**Missing API Key:**
```
Error: CHART_IMG_API_KEY not found
Solution: Set environment variable
```

**Invalid Symbol:**
```
Error: Symbol not found
Solution: Use EXCHANGE:SYMBOL format
```

**Rate Limited:**
```
Error: Too many requests
Solution: Check Chart-IMG plan limits
```

## Development

### Setting up Development Environment
```bash
# Clone the repository
git clone <repository-url>
cd chart-mcp-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/

# Run all tests
pytest --cov=src/chart_mcp_python --cov-report=html
```

### Project Structure
```
chart-mcp-python/
├── src/
│   └── chart_mcp_python/
│       ├── __init__.py
│       ├── main.py              # STDIO MCP server
│       ├── http_server.py       # HTTP server
│       ├── sse_server.py        # SSE server
│       ├── server/
│       │   ├── __init__.py
│       │   ├── chart_service.py # Chart-IMG API service
│       │   └── storage.py       # In-memory storage
│       └── shared/
│           ├── __init__.py
│           └── schema.py        # Pydantic models
├── pyproject.toml
└── README.md
```

## Server Architecture

| Mode | Transport | Use Case | Real-time |
|------|-----------|----------|-----------|
| STDIO | Standard I/O | Claude Desktop | No |
| HTTP | HTTP REST | Universal clients | No |
| SSE | Server-Sent Events | Real-time apps | Yes |

## Migration from TypeScript

This Python implementation provides full feature parity with the original TypeScript version:

- **Compatible API**: Same tools and parameters
- **Enhanced Type Safety**: Pydantic models with validation
- **Better Performance**: FastMCP for efficient MCP handling
- **Real-time Features**: SSE support with event streaming
- **Modern Python**: Async/await, type hints, and modern practices

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

- Report issues on GitHub
- Check the [Chart-IMG API documentation](https://chart-img.com/docs)
- Review the MCP specification