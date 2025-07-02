"""Main MCP server using FastMCP."""

import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .server.chart_service import chart_service
from .server.storage import storage
from .shared.schema import (
    ChartConfig, ChartIndicator, InsertChartRequest, MovingAverageArgs, RSIArgs, MACDArgs,
    BollingerBandsArgs, StochasticArgs, ATRArgs, ADXArgs, AroonArgs, CCIArgs,
    CMFArgs, IchimokuArgs, ParabolicSARArgs, VolumeArgs, OBVArgs, WilliamsRArgs,
    UltimateOscillatorArgs, ROCArgs, MomentumArgs, MFIArgs, ArnaudLegoux_MAArgs,
    HullMAArgs, KeltnerChannelsArgs, DonchianChannelsArgs, LinearRegressionArgs,
    PriceChannelArgs, ChaikinOscillatorArgs, AwesomeOscillatorArgs,
    PriceOscillatorArgs, Detrended_PriceOscillatorArgs, CoppockCurveArgs,
    ConnorsRSIArgs, ChopZoneArgs, ChoppinessIndexArgs, ChandeKrollStopArgs,
    ChandeMomentumOscillatorArgs, BalanceOfPowerArgs, AccumulationDistributionArgs,
    AccumulativeSwingIndexArgs, AdvanceDeclineArgs, EaseOfMovementArgs,
    ElderForceIndexArgs, EnvelopesArgs, FisherTransformArgs, HistoricalVolatilityArgs,
    KlingerOscillatorArgs, KnowSureThingArgs, LeastSquaresMAArgs,
    LinearRegressionSlopeArgs, MACrossArgs, MAWithEMACrossArgs, MajorityRuleArgs,
    MassIndexArgs, McGinleyDynamicArgs, MovingAverageChannelArgs,
    MovingAverageDoubleArgs, MovingAverageHammingArgs, MovingAverageMultipleArgs,
    MovingAverageTripleArgs, NetVolumeArgs, PriceVolumeTrendArgs,
    RelativeVigorIndexArgs, SMIErgodicArgs, DirectionalMovementArgs,
    ChaikinVolatilityArgs
)


class InitializeChartArgs(BaseModel):
    """Arguments for initialize_chart tool."""
    symbol: str = Field(..., description="Trading symbol in format EXCHANGE:SYMBOL (e.g., NASDAQ:AAPL)")
    interval: str = Field(..., description="Chart time interval")
    chartType: str = Field(..., description="Type of chart to generate")
    duration: str = Field(default="1M", description="Chart duration/timeframe")
    width: int = Field(default=800, description="Chart width in pixels")
    height: int = Field(default=600, description="Chart height in pixels")
    theme: str = Field(default="light", description="Chart theme")
    indicators: List[Dict[str, Any]] = Field(default_factory=list, description="Technical indicators")
    drawings: List[Dict[str, Any]] = Field(default_factory=list, description="Chart drawings")
    showVolume: bool = Field(default=True, description="Show volume indicator")
    showGrid: bool = Field(default=True, description="Show chart grid")
    timezone: str = Field(default="America/New_York", description="Chart timezone")


class DownloadChartArgs(BaseModel):
    """Arguments for download_chart tool."""
    requestId: str = Field(..., description="The request ID returned from generate_chart")


class AddIndicatorArgs(BaseModel):
    """Arguments for adding indicators to existing chart request."""
    requestId: str = Field(..., description="The request ID to add indicator to")


class GetChartStatusArgs(BaseModel):
    """Arguments for get_chart_status tool."""
    requestId: str = Field(..., description="The request ID returned from generate_chart")


class GetRecentRequestsArgs(BaseModel):
    """Arguments for get_recent_requests tool."""
    limit: int = Field(default=10, description="Maximum number of requests to return")


# Initialize FastMCP
mcp = FastMCP("Chart-IMG MCP Server")


@mcp.tool()
async def initialize_chart(args: InitializeChartArgs) -> str:
    """Create a chart request and store parameters without generating the chart yet."""
    try:
        # Create chart request with initial parameters
        insert_request = InsertChartRequest(
            symbol=args.symbol,
            interval=args.interval,
            chartType=args.chartType,
            duration=args.duration,
            width=args.width,
            height=args.height,
            indicators=args.indicators,
            drawings=args.drawings,
            theme=args.theme,
            showVolume=args.showVolume,
            showGrid=args.showGrid,
            timezone=args.timezone,
        )
        
        chart_request = await storage.create_chart_request(insert_request)

        response_text = (
            f"Chart request created successfully!\n\n"
            f"Request ID: {chart_request.requestId}\n"
            f"Symbol: {args.symbol}\n"
            f"Interval: {args.interval}\n"
            f"Chart Type: {args.chartType}\n"
            f"Duration: {args.duration}\n\n"
            f"You can now add indicators using tools like add_rsi, then call download_chart to generate the final chart."
        )
        
        return response_text

    except Exception as e:
        return f"Error creating chart request: {str(e)}"


@mcp.tool()
async def download_chart(args: DownloadChartArgs) -> str:
    """Generate and download the final chart with all stored parameters and indicators."""
    try:
        chart_request = await storage.get_chart_request(args.requestId)
        if not chart_request:
            return f"Chart request {args.requestId} not found"
        
        # Get stored indicators and pass them directly to chart service
        stored_indicators = chart_request.indicators or []
        
        # Build chart configuration from stored request (without indicators in config)
        config = ChartConfig(
            symbol=chart_request.symbol,
            interval=chart_request.interval,
            chartType=chart_request.chartType,
            duration=chart_request.duration,
            width=chart_request.width,
            height=chart_request.height,
            theme=chart_request.theme,
            indicators=[],  # Leave empty, we'll pass raw indicators separately
            drawings=chart_request.drawings or [],
            showVolume=chart_request.showVolume,
            showGrid=chart_request.showGrid,
            timezone=chart_request.timezone,
        )

        # Build the chart payload for transparency
        chart_payload = chart_service._build_chart_payload(config, stored_indicators)
        
        # Initialize the chart with raw indicators
        result = await chart_service.initialize_chart(config, chart_request.requestId, stored_indicators)

        # Update the chart request with the result
        if result.success:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "completed",
                "chartUrl": result.url,
                "base64Data": result.base64,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            import json
            payload_json = json.dumps(chart_payload, indent=2)
            
            response_text = (
                f"Chart generated successfully!\n\n"
                f"Request ID: {chart_request.requestId}\n"
                f"Symbol: {config.symbol}\n"
                f"Interval: {config.interval}\n"
                f"Chart Type: {config.chartType}\n"
                f"Duration: {config.duration}\n"
                f"Indicators: {len(chart_request.indicators or [])}\n"
                f"Processing Time: {(result.processingTime or 0) / 1000:.1f}s\n\n"
                f"JSON payload sent to chart-img API:\n```json\n{payload_json}\n```\n\n"
                f"The chart has been generated and is available as a PNG image."
            )

            if result.base64:
                return f"{response_text}\n[Base64 Image Data Available]"
            
            return response_text

        else:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "failed",
                "errorMessage": result.error,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            raise Exception(f"Chart initialization failed: {result.error}")

    except Exception as e:
        return f"Error initializing chart: {str(e)}"


@mcp.tool()
async def get_chart_status(args: GetChartStatusArgs) -> str:
    """Check the status of a chart generation request."""
    try:
        chart_request = await storage.get_chart_request(args.requestId)
        
        if not chart_request:
            return "Chart request not found"

        status_parts = [
            "Chart Status Report",
            "",
            f"Request ID: {chart_request.requestId}",
            f"Symbol: {chart_request.symbol}",
            f"Interval: {chart_request.interval}",
            f"Chart Type: {chart_request.chartType}",
            f"Duration: {chart_request.duration}",
            f"Status: {chart_request.status.upper()}",
            f"Created: {chart_request.createdAt.isoformat()}",
        ]

        if chart_request.completedAt:
            status_parts.append(f"Completed: {chart_request.completedAt.isoformat()}")
        
        if chart_request.processingTime:
            status_parts.append(f"Processing Time: {chart_request.processingTime / 1000:.1f}s")
        
        if chart_request.errorMessage:
            status_parts.append(f"Error: {chart_request.errorMessage}")

        if chart_request.status == "completed" and chart_request.base64Data:
            status_parts.append("[Chart image data available]")

        return "\n".join(status_parts)

    except Exception as e:
        return f"Error getting chart status: {str(e)}"


@mcp.tool()
async def get_available_symbols() -> str:
    """Get list of available trading symbols from Chart-IMG API."""
    try:
        symbols = await chart_service.get_available_symbols()
        
        return (
            f"Available Trading Symbols\n\n"
            f"Total symbols available: {len(symbols)}\n\n"
            f"Note: Use symbols in EXCHANGE:SYMBOL format (e.g., NASDAQ:AAPL, NYSE:TSLA, BINANCE:BTCUSDT)\n\n"
            f"Common exchanges:\n"
            f"- NASDAQ: US tech stocks\n"
            f"- NYSE: US stocks\n"
            f"- BINANCE: Crypto pairs\n"
            f"- FOREX: Currency pairs\n"
            f"- CRYPTO: Cryptocurrency symbols"
        )

    except Exception as e:
        return f"Error getting available symbols: {str(e)}"


@mcp.tool()
async def get_recent_requests(args: GetRecentRequestsArgs) -> str:
    """Get recent chart generation requests with their status."""
    try:
        requests = await storage.get_recent_chart_requests(args.limit)

        if not requests:
            return "No recent requests found."

        lines = [
            f"Recent Chart Requests ({len(requests)} of {args.limit})",
            "",
            "Format: Request ID | Symbol | Interval | Type | Duration | Status | Created",
            "",
        ]

        for req in requests:
            line = (
                f"{req.requestId} | {req.symbol} | {req.interval} | "
                f"{req.chartType} | {req.duration} | {req.status.upper()} | "
                f"{req.createdAt.isoformat()}"
            )
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting recent requests: {str(e)}"


class ClearChartArgs(BaseModel):
    """Arguments for clear_chart tool."""
    requestId: str = Field(..., description="The request ID of the chart to clear")


@mcp.tool()
async def clear_chart(args: ClearChartArgs) -> str:
    """Clear a specific chart request and its related data from storage."""
    try:
        success = await storage.delete_chart_request(args.requestId)
        
        if success:
            return f"Chart request {args.requestId} and its related data cleared successfully."
        else:
            return f"Chart request {args.requestId} not found in storage."
    except Exception as e:
        return f"Error clearing chart request: {str(e)}"


@mcp.tool()
async def health_check() -> str:
    """Check the health and configuration status of the chart service."""
    try:
        is_configured = chart_service.is_configured()
        
        status_parts = [
            "Chart Service Health Check",
            "",
            f"Service Status: {'HEALTHY' if is_configured else 'CONFIGURATION_ERROR'}",
            f"Chart-IMG API: {'CONFIGURED' if is_configured else 'API_KEY_MISSING'}",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
        ]

        if not is_configured:
            status_parts.append(
                "Please set the CHART_IMG_API_KEY environment variable to use the chart generation service."
            )
        else:
            status_parts.append("All systems operational. Ready to generate charts.")

        return "\n".join(status_parts)

    except Exception as e:
        return f"Error performing health check: {str(e)}"


# ============ MOVING AVERAGE INDICATORS ============

@mcp.tool()
async def add_sma(args: MovingAverageArgs) -> str:
    """Add Simple Moving Average indicator to existing chart request."""
    try:
        # Validate chart request exists
        error_msg = await validate_chart_request(args.requestId)
        if error_msg:
            return error_msg
        
        # Get the existing chart request
        chart_request = await storage.get_chart_request(args.requestId)
        
        # Build indicator config
        indicator_config = chart_service.add_indicator_to_chart("add_sma", args)
        
        # Add to existing indicators
        current_indicators = list(chart_request.indicators or [])
        current_indicators.append(indicator_config)
        
        # Update the chart request
        await storage.update_chart_request(args.requestId, {
            "indicators": current_indicators
        })
        
        return f"Simple Moving Average added to chart request {args.requestId} (Length: {args.length})"
    except Exception as e:
        return f"Error adding SMA: {str(e)}"


@mcp.tool()
async def add_ema(args: MovingAverageArgs) -> str:
    """Add Exponential Moving Average indicator to existing chart request."""
    try:
        # Validate chart request exists
        error_msg = await validate_chart_request(args.requestId)
        if error_msg:
            return error_msg
        
        # Get the existing chart request
        chart_request = await storage.get_chart_request(args.requestId)
        
        # Build indicator config
        indicator_config = chart_service.add_indicator_to_chart("add_ema", args)
        
        # Add to existing indicators
        current_indicators = list(chart_request.indicators or [])
        current_indicators.append(indicator_config)
        
        # Update the chart request
        await storage.update_chart_request(args.requestId, {
            "indicators": current_indicators
        })
        
        return f"Exponential Moving Average added to chart request {args.requestId} (Length: {args.length})"
    except Exception as e:
        return f"Error adding EMA: {str(e)}"


@mcp.tool()
async def add_wma(args: MovingAverageArgs) -> str:
    """Add Weighted Moving Average indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_wma", args)
        return f"Weighted Moving Average added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding WMA: {str(e)}"


@mcp.tool()
async def add_hull_ma(args: HullMAArgs) -> str:
    """Add Hull Moving Average indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_hull_ma", args)
        return f"Hull Moving Average added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Hull MA: {str(e)}"


@mcp.tool()
async def add_arnaud_legoux_ma(args: ArnaudLegoux_MAArgs) -> str:
    """Add Arnaud Legoux Moving Average indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_arnaud_legoux_ma", args)
        return f"Arnaud Legoux Moving Average added to {args.symbol} (Length: {args.length}, Offset: {args.offset}, Sigma: {args.sigma})"
    except Exception as e:
        return f"Error adding Arnaud Legoux MA: {str(e)}"


@mcp.tool()
async def add_least_squares_ma(args: LeastSquaresMAArgs) -> str:
    """Add Least Squares Moving Average indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_least_squares_ma", args)
        return f"Least Squares Moving Average added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Least Squares MA: {str(e)}"


@mcp.tool()
async def add_mcginley_dynamic(args: McGinleyDynamicArgs) -> str:
    """Add McGinley Dynamic indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_mcginley_dynamic", args)
        return f"McGinley Dynamic added to {args.symbol} (Length: {args.length}, Constant: {args.constant})"
    except Exception as e:
        return f"Error adding McGinley Dynamic: {str(e)}"


@mcp.tool()
async def add_ma_channel(args: MovingAverageChannelArgs) -> str:
    """Add Moving Average Channel indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ma_channel", args)
        return f"Moving Average Channel added to {args.symbol} (Length: {args.length}, Percent: {args.percent}%)"
    except Exception as e:
        return f"Error adding MA Channel: {str(e)}"


@mcp.tool()
async def add_ma_multiple(args: MovingAverageMultipleArgs) -> str:
    """Add Moving Average Multiple indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ma_multiple", args)
        return f"Moving Average Multiple added to {args.symbol} (Lengths: {args.length1}, {args.length2}, {args.length3})"
    except Exception as e:
        return f"Error adding MA Multiple: {str(e)}"


@mcp.tool()
async def add_ma_hamming(args: MovingAverageHammingArgs) -> str:
    """Add Moving Average Hamming indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ma_hamming", args)
        return f"Moving Average Hamming added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding MA Hamming: {str(e)}"


@mcp.tool()
async def add_double_ema(args: MovingAverageDoubleArgs) -> str:
    """Add Double EMA indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_double_ema", args)
        return f"Double EMA added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Double EMA: {str(e)}"


@mcp.tool()
async def add_triple_ma(args: MovingAverageTripleArgs) -> str:
    """Add Triple Moving Average indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_triple_ma", args)
        return f"Triple Moving Average added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Triple MA: {str(e)}"


# ============ OSCILLATOR INDICATORS ============

async def validate_chart_request(request_id: str) -> str:
    """Validate that a chart request exists. Returns error message if invalid, empty string if valid."""
    if not request_id:
        return "Error: No request ID provided. Please call initialize_chart first to initialize a chart request."
    
    chart_request = await storage.get_chart_request(request_id)
    if not chart_request:
        return f"Error: Chart request {request_id} not found. Please call initialize_chart first to initialize a chart request."
    
    return ""


@mcp.tool()
async def add_rsi(args: RSIArgs) -> str:
    """Add Relative Strength Index indicator to existing chart request."""
    try:
        # Validate chart request exists
        error_msg = await validate_chart_request(args.requestId)
        if error_msg:
            return error_msg
        
        # Get the existing chart request
        chart_request = await storage.get_chart_request(args.requestId)
        
        # Build indicator config
        indicator_config = chart_service.add_indicator_to_chart("add_rsi", args)
        
        # Add to existing indicators
        current_indicators = list(chart_request.indicators or [])
        current_indicators.append(indicator_config)
        
        # Update the chart request
        await storage.update_chart_request(args.requestId, {
            "indicators": current_indicators
        })
        
        return f"RSI indicator added to chart request {args.requestId} (Length: {args.length})"
    except Exception as e:
        return f"Error adding RSI: {str(e)}"


@mcp.tool()
async def add_macd(args: MACDArgs) -> str:
    """Add MACD indicator to existing chart request."""
    try:
        # Validate chart request exists
        error_msg = await validate_chart_request(args.requestId)
        if error_msg:
            return error_msg
        
        # Get the existing chart request
        chart_request = await storage.get_chart_request(args.requestId)
        
        # Build indicator config
        indicator_config = chart_service.add_indicator_to_chart("add_macd", args)
        
        # Add to existing indicators
        current_indicators = list(chart_request.indicators or [])
        current_indicators.append(indicator_config)
        
        # Update the chart request
        await storage.update_chart_request(args.requestId, {
            "indicators": current_indicators
        })
        
        return f"MACD added to chart request {args.requestId} (Fast: {args.fast_length}, Slow: {args.slow_length}, Signal: {args.signal_length})"
    except Exception as e:
        return f"Error adding MACD: {str(e)}"


@mcp.tool()
async def add_stochastic(args: StochasticArgs) -> str:
    """Add Stochastic oscillator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_stochastic", args)
        return f"Stochastic added to {args.symbol} (%K: {args.k_length}, Smoothing: {args.k_smoothing}, %D: {args.d_length})"
    except Exception as e:
        return f"Error adding Stochastic: {str(e)}"


@mcp.tool()
async def add_cci(args: CCIArgs) -> str:
    """Add Commodity Channel Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_cci", args)
        return f"CCI added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding CCI: {str(e)}"


@mcp.tool()
async def add_williams_r(args: WilliamsRArgs) -> str:
    """Add Williams %R indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_williams_r", args)
        return f"Williams %R added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Williams %R: {str(e)}"


@mcp.tool()
async def add_momentum(args: MomentumArgs) -> str:
    """Add Momentum indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_momentum", args)
        return f"Momentum added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Momentum: {str(e)}"


@mcp.tool()
async def add_roc(args: ROCArgs) -> str:
    """Add Rate of Change indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_roc", args)
        return f"Rate of Change added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding ROC: {str(e)}"


@mcp.tool()
async def add_mfi(args: MFIArgs) -> str:
    """Add Money Flow Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_mfi", args)
        return f"Money Flow Index added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding MFI: {str(e)}"


@mcp.tool()
async def add_ultimate_oscillator(args: UltimateOscillatorArgs) -> str:
    """Add Ultimate Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ultimate_oscillator", args)
        return f"Ultimate Oscillator added to {args.symbol} (Periods: {args.length1}, {args.length2}, {args.length3})"
    except Exception as e:
        return f"Error adding Ultimate Oscillator: {str(e)}"


@mcp.tool()
async def add_awesome_oscillator(args: AwesomeOscillatorArgs) -> str:
    """Add Awesome Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_awesome_oscillator", args)
        return f"Awesome Oscillator added to {args.symbol} (Fast: {args.fast_length}, Slow: {args.slow_length})"
    except Exception as e:
        return f"Error adding Awesome Oscillator: {str(e)}"


@mcp.tool()
async def add_price_oscillator(args: PriceOscillatorArgs) -> str:
    """Add Price Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_price_oscillator", args)
        return f"Price Oscillator added to {args.symbol} (Fast: {args.fast_length}, Slow: {args.slow_length}, Signal: {args.signal_length})"
    except Exception as e:
        return f"Error adding Price Oscillator: {str(e)}"


@mcp.tool()
async def add_detrended_price_oscillator(args: Detrended_PriceOscillatorArgs) -> str:
    """Add Detrended Price Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_detrended_price_oscillator", args)
        return f"Detrended Price Oscillator added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Detrended Price Oscillator: {str(e)}"


@mcp.tool()
async def add_chaikin_oscillator(args: ChaikinOscillatorArgs) -> str:
    """Add Chaikin Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_chaikin_oscillator", args)
        return f"Chaikin Oscillator added to {args.symbol} (Fast: {args.fast_length}, Slow: {args.slow_length})"
    except Exception as e:
        return f"Error adding Chaikin Oscillator: {str(e)}"


@mcp.tool()
async def add_klinger_oscillator(args: KlingerOscillatorArgs) -> str:
    """Add Klinger Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_klinger_oscillator", args)
        return f"Klinger Oscillator added to {args.symbol} (Fast: {args.fast_length}, Slow: {args.slow_length}, Signal: {args.signal_length})"
    except Exception as e:
        return f"Error adding Klinger Oscillator: {str(e)}"


@mcp.tool()
async def add_connors_rsi(args: ConnorsRSIArgs) -> str:
    """Add Connors RSI indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_connors_rsi", args)
        return f"Connors RSI added to {args.symbol} (RSI: {args.rsi_length}, Streak: {args.streak_length}, Rank: {args.rank_length})"
    except Exception as e:
        return f"Error adding Connors RSI: {str(e)}"


@mcp.tool()
async def add_chande_momentum_oscillator(args: ChandeMomentumOscillatorArgs) -> str:
    """Add Chande Momentum Oscillator indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_chande_momentum_oscillator", args)
        return f"Chande Momentum Oscillator added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Chande Momentum Oscillator: {str(e)}"


@mcp.tool()
async def add_relative_vigor_index(args: RelativeVigorIndexArgs) -> str:
    """Add Relative Vigor Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_relative_vigor_index", args)
        return f"Relative Vigor Index added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Relative Vigor Index: {str(e)}"


@mcp.tool()
async def add_smi_ergodic(args: SMIErgodicArgs) -> str:
    """Add SMI Ergodic indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_smi_ergodic", args)
        return f"SMI Ergodic added to {args.symbol} (Length1: {args.length1}, Length2: {args.length2}, Signal: {args.signal_length})"
    except Exception as e:
        return f"Error adding SMI Ergodic: {str(e)}"


# ============ VOLUME INDICATORS ============

@mcp.tool()
async def add_volume(args: VolumeArgs) -> str:
    """Add Volume indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_volume", args)
        ma_text = f" with MA({args.ma_length})" if args.show_ma else ""
        return f"Volume added to {args.symbol}{ma_text}"
    except Exception as e:
        return f"Error adding Volume: {str(e)}"


@mcp.tool()
async def add_obv(args: OBVArgs) -> str:
    """Add On Balance Volume indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_obv", args)
        return f"On Balance Volume added to {args.symbol}"
    except Exception as e:
        return f"Error adding OBV: {str(e)}"


@mcp.tool()
async def add_cmf(args: CMFArgs) -> str:
    """Add Chaikin Money Flow indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_cmf", args)
        return f"Chaikin Money Flow added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding CMF: {str(e)}"


@mcp.tool()
async def add_ease_of_movement(args: EaseOfMovementArgs) -> str:
    """Add Ease of Movement indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ease_of_movement", args)
        return f"Ease of Movement added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Ease of Movement: {str(e)}"


@mcp.tool()
async def add_accumulation_distribution(args: AccumulationDistributionArgs) -> str:
    """Add Accumulation/Distribution indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_accumulation_distribution", args)
        return f"Accumulation/Distribution added to {args.symbol}"
    except Exception as e:
        return f"Error adding Accumulation/Distribution: {str(e)}"


@mcp.tool()
async def add_elder_force_index(args: ElderForceIndexArgs) -> str:
    """Add Elder's Force Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_elder_force_index", args)
        return f"Elder's Force Index added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Elder's Force Index: {str(e)}"


@mcp.tool()
async def add_net_volume(args: NetVolumeArgs) -> str:
    """Add Net Volume indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_net_volume", args)
        return f"Net Volume added to {args.symbol}"
    except Exception as e:
        return f"Error adding Net Volume: {str(e)}"


@mcp.tool()
async def add_price_volume_trend(args: PriceVolumeTrendArgs) -> str:
    """Add Price Volume Trend indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_price_volume_trend", args)
        return f"Price Volume Trend added to {args.symbol}"
    except Exception as e:
        return f"Error adding Price Volume Trend: {str(e)}"


# ============ VOLATILITY INDICATORS ============

@mcp.tool()
async def add_bollinger_bands(args: BollingerBandsArgs) -> str:
    """Add Bollinger Bands indicator to existing chart request."""
    try:
        # Validate chart request exists
        error_msg = await validate_chart_request(args.requestId)
        if error_msg:
            return error_msg
        
        # Get the existing chart request
        chart_request = await storage.get_chart_request(args.requestId)
        
        # Build indicator config
        indicator_config = chart_service.add_indicator_to_chart("add_bollinger_bands", args)
        
        # Add to existing indicators
        current_indicators = list(chart_request.indicators or [])
        current_indicators.append(indicator_config)
        
        # Update the chart request
        await storage.update_chart_request(args.requestId, {
            "indicators": current_indicators
        })
        
        return f"Bollinger Bands added to chart request {args.requestId} (Length: {args.length}, Multiplier: {args.mult})"
    except Exception as e:
        return f"Error adding Bollinger Bands: {str(e)}"


@mcp.tool()
async def add_atr(args: ATRArgs) -> str:
    """Add Average True Range indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_atr", args)
        return f"Average True Range added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding ATR: {str(e)}"


@mcp.tool()
async def add_keltner_channels(args: KeltnerChannelsArgs) -> str:
    """Add Keltner Channels indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_keltner_channels", args)
        return f"Keltner Channels added to {args.symbol} (EMA: {args.length}, ATR Mult: {args.mult}, ATR: {args.atr_length})"
    except Exception as e:
        return f"Error adding Keltner Channels: {str(e)}"


@mcp.tool()
async def add_donchian_channels(args: DonchianChannelsArgs) -> str:
    """Add Donchian Channels indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_donchian_channels", args)
        return f"Donchian Channels added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Donchian Channels: {str(e)}"


@mcp.tool()
async def add_historical_volatility(args: HistoricalVolatilityArgs) -> str:
    """Add Historical Volatility indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_historical_volatility", args)
        return f"Historical Volatility added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Historical Volatility: {str(e)}"


@mcp.tool()
async def add_envelopes(args: EnvelopesArgs) -> str:
    """Add Envelopes indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_envelopes", args)
        return f"Envelopes added to {args.symbol} (Length: {args.length}, Percent: {args.percent}%)"
    except Exception as e:
        return f"Error adding Envelopes: {str(e)}"


@mcp.tool()
async def add_chaikin_volatility(args: ChaikinVolatilityArgs) -> str:
    """Add Chaikin Volatility indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_chaikin_volatility", args)
        return f"Chaikin Volatility added to {args.symbol} (EMA: {args.ema_length}, ROC: {args.roc_length})"
    except Exception as e:
        return f"Error adding Chaikin Volatility: {str(e)}"


# ============ TREND INDICATORS ============

@mcp.tool()
async def add_parabolic_sar(args: ParabolicSARArgs) -> str:
    """Add Parabolic SAR indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_parabolic_sar", args)
        return f"Parabolic SAR added to {args.symbol} (Start: {args.start}, Increment: {args.increment}, Max: {args.maximum})"
    except Exception as e:
        return f"Error adding Parabolic SAR: {str(e)}"


@mcp.tool()
async def add_adx(args: ADXArgs) -> str:
    """Add Average Directional Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_adx", args)
        return f"Average Directional Index added to {args.symbol} (Smoothing: {args.smoothing}, DI: {args.di_length})"
    except Exception as e:
        return f"Error adding ADX: {str(e)}"


@mcp.tool()
async def add_aroon(args: AroonArgs) -> str:
    """Add Aroon indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_aroon", args)
        return f"Aroon added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Aroon: {str(e)}"


@mcp.tool()
async def add_ichimoku(args: IchimokuArgs) -> str:
    """Add Ichimoku Cloud indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ichimoku", args)
        return f"Ichimoku Cloud added to {args.symbol} (Conversion: {args.conversion_line_length}, Base: {args.base_line_length}, Span B: {args.leading_span_b_length}, Displacement: {args.displacement})"
    except Exception as e:
        return f"Error adding Ichimoku: {str(e)}"


@mcp.tool()
async def add_linear_regression(args: LinearRegressionArgs) -> str:
    """Add Linear Regression indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_linear_regression", args)
        return f"Linear Regression added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Linear Regression: {str(e)}"


@mcp.tool()
async def add_linear_regression_slope(args: LinearRegressionSlopeArgs) -> str:
    """Add Linear Regression Slope indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_linear_regression_slope", args)
        return f"Linear Regression Slope added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Linear Regression Slope: {str(e)}"


@mcp.tool()
async def add_directional_movement(args: DirectionalMovementArgs) -> str:
    """Add Directional Movement indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_directional_movement", args)
        return f"Directional Movement added to {args.symbol} (ADX: {args.adx_smoothing}, DI: {args.di_length})"
    except Exception as e:
        return f"Error adding Directional Movement: {str(e)}"


@mcp.tool()
async def add_ma_cross(args: MACrossArgs) -> str:
    """Add MA Cross indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ma_cross", args)
        return f"MA Cross added to {args.symbol} (Fast: {args.fast_length}, Slow: {args.slow_length})"
    except Exception as e:
        return f"Error adding MA Cross: {str(e)}"


@mcp.tool()
async def add_ma_with_ema_cross(args: MAWithEMACrossArgs) -> str:
    """Add MA with EMA Cross indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_ma_with_ema_cross", args)
        return f"MA with EMA Cross added to {args.symbol} (MA: {args.ma_length}, EMA: {args.ema_length})"
    except Exception as e:
        return f"Error adding MA with EMA Cross: {str(e)}"


@mcp.tool()
async def add_coppock_curve(args: CoppockCurveArgs) -> str:
    """Add Coppock Curve indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_coppock_curve", args)
        return f"Coppock Curve added to {args.symbol} (WMA: {args.wma_length}, ROC1: {args.roc1_length}, ROC2: {args.roc2_length})"
    except Exception as e:
        return f"Error adding Coppock Curve: {str(e)}"


# ============ OTHER INDICATORS ============

@mcp.tool()
async def add_balance_of_power(args: BalanceOfPowerArgs) -> str:
    """Add Balance of Power indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_balance_of_power", args)
        return f"Balance of Power added to {args.symbol}"
    except Exception as e:
        return f"Error adding Balance of Power: {str(e)}"


@mcp.tool()
async def add_accumulative_swing_index(args: AccumulativeSwingIndexArgs) -> str:
    """Add Accumulative Swing Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_accumulative_swing_index", args)
        return f"Accumulative Swing Index added to {args.symbol} (Limit Move: {args.limit_move})"
    except Exception as e:
        return f"Error adding Accumulative Swing Index: {str(e)}"


@mcp.tool()
async def add_advance_decline(args: AdvanceDeclineArgs) -> str:
    """Add Advance/Decline indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_advance_decline", args)
        return f"Advance/Decline added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Advance/Decline: {str(e)}"


@mcp.tool()
async def add_fisher_transform(args: FisherTransformArgs) -> str:
    """Add Fisher Transform indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_fisher_transform", args)
        return f"Fisher Transform added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Fisher Transform: {str(e)}"


@mcp.tool()
async def add_know_sure_thing(args: KnowSureThingArgs) -> str:
    """Add Know Sure Thing indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_know_sure_thing", args)
        return f"Know Sure Thing added to {args.symbol} (ROC: {args.roc1_length},{args.roc2_length},{args.roc3_length},{args.roc4_length}, SMA: {args.sma1_length},{args.sma2_length},{args.sma3_length},{args.sma4_length})"
    except Exception as e:
        return f"Error adding Know Sure Thing: {str(e)}"


@mcp.tool()
async def add_price_channel(args: PriceChannelArgs) -> str:
    """Add Price Channel indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_price_channel", args)
        return f"Price Channel added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Price Channel: {str(e)}"


@mcp.tool()
async def add_chop_zone(args: ChopZoneArgs) -> str:
    """Add Chop Zone indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_chop_zone", args)
        return f"Chop Zone added to {args.symbol} (EMA: {args.ema_length})"
    except Exception as e:
        return f"Error adding Chop Zone: {str(e)}"


@mcp.tool()
async def add_choppiness_index(args: ChoppinessIndexArgs) -> str:
    """Add Choppiness Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_choppiness_index", args)
        return f"Choppiness Index added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Choppiness Index: {str(e)}"


@mcp.tool()
async def add_chande_kroll_stop(args: ChandeKrollStopArgs) -> str:
    """Add Chande Kroll Stop indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_chande_kroll_stop", args)
        return f"Chande Kroll Stop added to {args.symbol} (Length: {args.length}, Multiplier: {args.mult})"
    except Exception as e:
        return f"Error adding Chande Kroll Stop: {str(e)}"


@mcp.tool()
async def add_majority_rule(args: MajorityRuleArgs) -> str:
    """Add Majority Rule indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_majority_rule", args)
        return f"Majority Rule added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Majority Rule: {str(e)}"


@mcp.tool()
async def add_mass_index(args: MassIndexArgs) -> str:
    """Add Mass Index indicator to chart."""
    try:
        indicator_config = chart_service.add_indicator_to_chart("add_mass_index", args)
        return f"Mass Index added to {args.symbol} (Length: {args.length})"
    except Exception as e:
        return f"Error adding Mass Index: {str(e)}"


def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\nShutting down Chart-IMG MCP Server", file=sys.stderr)
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()