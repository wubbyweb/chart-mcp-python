"""Chart service for generating charts using Chart-IMG API."""

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from ..shared.schema import ChartConfig, ChartImgResponse
from .indicator_service import indicator_service


class ChartService:
    """Service for interacting with Chart-IMG API."""

    def __init__(self) -> None:
        """Initialize the chart service."""
        self.api_key = os.getenv("CHART_IMG_API_KEY") or os.getenv("API_KEY") or ""
        self.base_url = "https://api.chart-img.com"
        self.output_dir = Path("./output")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.api_key:
            print("Warning: CHART_IMG_API_KEY not found in environment variables")

    async def initialize_chart(self, config: ChartConfig, request_id: str, raw_indicators: List[Dict[str, Any]] = None) -> ChartImgResponse:
        """Generate a chart using the Chart-IMG API."""
        start_time = int(time.time() * 1000)

        try:
            payload = self._build_chart_payload(config, raw_indicators)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v2/tradingview/advanced-chart",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    timeout=30.0,
                )

            if not response.is_success:
                error_text = response.text
                raise Exception(f"Chart-IMG API error: {response.status_code} {error_text}")

            content_type = response.headers.get("content-type", "")
            processing_time = int(time.time() * 1000) - start_time

            if "application/json" in content_type:
                json_data = response.json()
                base64_data = json_data.get("base64", "")
                
                # Strip data URL prefix if present for MCP compatibility
                if base64_data and base64_data.startswith("data:image/"):
                    base64_data = base64_data.split(",", 1)[1]
                
                # Save PNG to output directory
                png_path = self._save_png(base64_data, request_id, config.symbol)
                
                return ChartImgResponse(
                    success=True,
                    url=json_data.get("url"),
                    base64=base64_data,
                    processingTime=processing_time,
                )
            elif "image/png" in content_type:
                # Direct PNG response
                base64_data = base64.b64encode(response.content).decode("utf-8")
                
                # Save PNG to output directory
                png_path = self._save_png(base64_data, request_id, config.symbol)
                
                return ChartImgResponse(
                    success=True,
                    base64=base64_data,
                    processingTime=processing_time,
                )
            else:
                raise Exception("Unexpected response format from Chart-IMG API")

        except Exception as e:
            error_message = str(e)
            processing_time = int(time.time() * 1000) - start_time
            
            return ChartImgResponse(
                success=False,
                error=error_message,
                processingTime=processing_time,
            )

    def _save_png(self, base64_data: str, request_id: str, symbol: str) -> Path:
        """Save base64 PNG data to output directory."""
        if not base64_data:
            raise ValueError("No base64 data provided")
        
        # Create safe filename from symbol and request_id
        safe_symbol = symbol.replace(":", "_").replace("/", "_")
        filename = f"chart_{safe_symbol}_{request_id}.png"
        filepath = self.output_dir / filename
        
        try:
            # Decode base64 and save as PNG
            png_data = base64.b64decode(base64_data)
            with open(filepath, 'wb') as f:
                f.write(png_data)
            
            print(f"Chart saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving PNG file: {e}")
            raise

    def _build_chart_payload(self, config: ChartConfig, raw_indicators: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the payload for Chart-IMG API."""
        payload: Dict[str, Any] = {
            "symbol": config.symbol,
            "interval": config.interval,
            "duration": config.duration,
            "width": config.width,
            "height": config.height,
            "theme": config.theme,
            "studies": [],
            "drawings": [],
        }

        # Convert chart type to Chart-IMG format
        chart_type_map = {
            "candlestick": 1,
            "line": 2,
            "area": 3,
            "bar": 0,
            "heikin_ashi": 9,
            "hollow_candle": 8,
            "baseline": 7,
            "hi_lo": 12,
            "column": 11,
        }
        payload["type"] = chart_type_map.get(config.chartType, 1)

        # Add indicators with correct Chart-IMG format
        if raw_indicators:
            # Use the raw indicators directly (they're already in Chart-IMG format)
            payload["studies"] = raw_indicators
        elif config.indicators:
            # Fallback to old method for backward compatibility
            payload["studies"] = []
            for indicator in config.indicators:
                study: Dict[str, Any] = {
                    "name": self._map_indicator_name(indicator.type),
                    "inputs": [],
                }

                # Add period input for indicators that support it
                if indicator.period:
                    study["inputs"].append(indicator.period)

                # Add style overrides if color is specified
                if indicator.color:
                    study["styles"] = {"plot_0": {"color": indicator.color}}

                payload["studies"].append(study)
        else:
            payload["studies"] = []

        # Add drawings with correct Chart-IMG format
        if config.drawings:
            payload["drawings"] = []
            for drawing in config.drawings:
                drawing_obj = {
                    "name": self._map_drawing_name(drawing.type),
                    "input": {
                        "points": [
                            {"time": point.x, "price": point.y} for point in drawing.points
                        ]
                    },
                    "options": {
                        "color": drawing.color or "#8B5CF6",
                        "linewidth": drawing.width or 2,
                    },
                }
                payload["drawings"].append(drawing_obj)

        # Additional options
        payload["options"] = {
            "timezone": config.timezone,
            "volume": config.showVolume,
            "grid": config.showGrid,
        }

        return payload

    def _map_indicator_name(self, indicator_type: str) -> str:
        """Map indicator type to Chart-IMG format."""
        # Legacy mapping for backward compatibility
        indicator_map = {
            "sma": "Moving Average",
            "ema": "Moving Average Exponential",
            "rsi": "Relative Strength Index",
            "macd": "MACD",
            "bb": "Bollinger Bands",
            "stoch": "Stochastic",
            "atr": "Average True Range",
            "volume": "Volume",
        }
        return indicator_map.get(indicator_type.lower(), "Moving Average")
    
    def add_indicator_to_chart(self, tool_name: str, args) -> Dict[str, Any]:
        """Add an indicator to chart using the new indicator service."""
        return indicator_service.build_indicator_config(tool_name, args)

    def _map_drawing_name(self, drawing_type: str) -> str:
        """Map drawing type to Chart-IMG format."""
        drawing_map = {
            "trendline": "Trend Line",
            "horizontal": "Horizontal Line",
            "vertical": "Vertical Line",
            "rectangle": "Rectangle",
        }
        return drawing_map.get(drawing_type.lower(), "Trend Line")

    async def get_available_symbols(self) -> List[str]:
        """Get available trading symbols from Chart-IMG API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v3/exchanges",
                    headers={"x-api-key": self.api_key},
                    timeout=10.0,
                )

            if not response.is_success:
                raise Exception(f"Failed to fetch symbols: {response.status_code}")

            data = response.json()
            return data.get("exchanges", [])

        except Exception as e:
            print(f"Error fetching symbols: {e}")
            # Return common symbol formats as fallback
            return [
                "NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:GOOGL", "NASDAQ:TSLA",
                "NYSE:IBM", "NYSE:JPM", "NYSE:KO", "NYSE:DIS",
                "BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:ADAUSDT",
                "FOREX:EURUSD", "FOREX:GBPUSD", "FOREX:USDJPY"
            ]

    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        return bool(self.api_key)


# Global instance
chart_service = ChartService()