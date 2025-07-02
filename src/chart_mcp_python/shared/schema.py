"""Schema definitions for Chart MCP Python."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator


class ChartIndicator(BaseModel):
    """Technical indicator configuration."""
    type: str = Field(..., description="Indicator type (e.g., sma, rsi, macd)")
    period: Optional[int] = Field(None, description="Period for the indicator")
    color: Optional[str] = Field(None, description="Color for the indicator")
    overbought: Optional[float] = Field(None, description="Overbought level for oscillators")
    oversold: Optional[float] = Field(None, description="Oversold level for oscillators")


class BaseIndicatorArgs(BaseModel):
    """Base arguments for indicator tools."""
    requestId: str = Field(..., description="Chart request ID to add indicator to")
    color: Optional[str] = Field(None, description="Indicator color")
    linewidth: Optional[int] = Field(None, ge=1, le=10, description="Line width (1-10)")
    plottype: Optional[str] = Field(None, description="Plot type (line, histogram, etc.)")


class MovingAverageArgs(BaseIndicatorArgs):
    """Arguments for moving average indicators."""
    length: int = Field(14, ge=1, le=5000, description="Period length")
    

class RSIArgs(BaseIndicatorArgs):
    """Arguments for RSI indicator."""
    length: int = Field(14, ge=1, le=5000, description="RSI period")


class MACDArgs(BaseIndicatorArgs):
    """Arguments for MACD indicator."""
    fast_length: int = Field(12, ge=1, le=1000, description="Fast EMA length")
    slow_length: int = Field(26, ge=1, le=1000, description="Slow EMA length")
    signal_length: int = Field(9, ge=1, le=1000, description="Signal line length")


class BollingerBandsArgs(BaseIndicatorArgs):
    """Arguments for Bollinger Bands indicator."""
    length: int = Field(20, ge=1, le=1000, description="Period length")
    mult: float = Field(2.0, ge=0.1, le=10.0, description="Standard deviation multiplier")


class StochasticArgs(BaseIndicatorArgs):
    """Arguments for Stochastic indicator."""
    k_length: int = Field(14, ge=1, le=1000, description="%K period")
    k_smoothing: int = Field(1, ge=1, le=100, description="%K smoothing")
    d_length: int = Field(3, ge=1, le=100, description="%D period")


class ATRArgs(BaseIndicatorArgs):
    """Arguments for Average True Range indicator."""
    length: int = Field(14, ge=1, le=1000, description="ATR period")


class ADXArgs(BaseIndicatorArgs):
    """Arguments for Average Directional Index indicator."""
    smoothing: int = Field(14, ge=1, le=1000, description="ADX smoothing period")
    di_length: int = Field(14, ge=1, le=1000, description="DI calculation length")


class AroonArgs(BaseIndicatorArgs):
    """Arguments for Aroon indicator."""
    length: int = Field(14, ge=1, le=2000, description="Aroon period")


class CCIArgs(BaseIndicatorArgs):
    """Arguments for Commodity Channel Index indicator."""
    length: int = Field(20, ge=1, le=1000, description="CCI period")


class CMFArgs(BaseIndicatorArgs):
    """Arguments for Chaikin Money Flow indicator."""
    length: int = Field(20, ge=1, le=1000, description="CMF period")


class IchimokuArgs(BaseIndicatorArgs):
    """Arguments for Ichimoku Cloud indicator."""
    conversion_line_length: int = Field(9, ge=1, le=1000, description="Conversion line period")
    base_line_length: int = Field(26, ge=1, le=1000, description="Base line period")
    leading_span_b_length: int = Field(52, ge=1, le=1000, description="Leading span B period")
    displacement: int = Field(26, ge=1, le=1000, description="Displacement")


class ParabolicSARArgs(BaseIndicatorArgs):
    """Arguments for Parabolic SAR indicator."""
    start: float = Field(0.02, ge=0.001, le=1.0, description="Start acceleration factor")
    increment: float = Field(0.02, ge=0.001, le=1.0, description="Increment")
    maximum: float = Field(0.2, ge=0.001, le=1.0, description="Maximum acceleration factor")


class FibonacciArgs(BaseIndicatorArgs):
    """Arguments for Fibonacci retracement indicator."""
    start_price: float = Field(..., description="Start price level")
    end_price: float = Field(..., description="End price level")
    
    
class VolumeArgs(BaseIndicatorArgs):
    """Arguments for Volume indicator."""
    show_ma: bool = Field(False, description="Show moving average")
    ma_length: int = Field(20, ge=1, le=1000, description="Moving average length")


class OBVArgs(BaseIndicatorArgs):
    """Arguments for On Balance Volume indicator."""
    pass  # OBV doesn't require additional parameters


class WilliamsRArgs(BaseIndicatorArgs):
    """Arguments for Williams %R indicator."""
    length: int = Field(14, ge=1, le=1000, description="Williams %R period")


class UltimateOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Ultimate Oscillator indicator."""
    length1: int = Field(7, ge=1, le=1000, description="First period")
    length2: int = Field(14, ge=1, le=1000, description="Second period")
    length3: int = Field(28, ge=1, le=1000, description="Third period")


class ROCArgs(BaseIndicatorArgs):
    """Arguments for Rate of Change indicator."""
    length: int = Field(10, ge=1, le=1000, description="ROC period")


class MomentumArgs(BaseIndicatorArgs):
    """Arguments for Momentum indicator."""
    length: int = Field(10, ge=1, le=1000, description="Momentum period")


class MFIArgs(BaseIndicatorArgs):
    """Arguments for Money Flow Index indicator."""
    length: int = Field(14, ge=1, le=1000, description="MFI period")


class ArnaudLegoux_MAArgs(BaseIndicatorArgs):
    """Arguments for Arnaud Legoux Moving Average indicator."""
    length: int = Field(21, ge=1, le=5000, description="Window size")
    offset: float = Field(0.0, ge=-100.0, le=100.0, description="Offset")
    sigma: float = Field(6.0, ge=0.1, le=100.0, description="Sigma")


class HullMAArgs(BaseIndicatorArgs):
    """Arguments for Hull Moving Average indicator."""
    length: int = Field(16, ge=1, le=5000, description="Hull MA period")


class KeltnerChannelsArgs(BaseIndicatorArgs):
    """Arguments for Keltner Channels indicator."""
    length: int = Field(20, ge=1, le=1000, description="EMA period")
    mult: float = Field(2.0, ge=0.1, le=10.0, description="ATR multiplier")
    atr_length: int = Field(10, ge=1, le=1000, description="ATR period")


class DonchianChannelsArgs(BaseIndicatorArgs):
    """Arguments for Donchian Channels indicator."""
    length: int = Field(20, ge=1, le=1000, description="Donchian period")


class LinearRegressionArgs(BaseIndicatorArgs):
    """Arguments for Linear Regression indicator."""
    length: int = Field(14, ge=1, le=1000, description="Linear regression period")


class PriceChannelArgs(BaseIndicatorArgs):
    """Arguments for Price Channel indicator."""
    length: int = Field(20, ge=1, le=1000, description="Price channel period")


class ChaikinOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Chaikin Oscillator indicator."""
    fast_length: int = Field(3, ge=1, le=1000, description="Fast EMA length")
    slow_length: int = Field(10, ge=1, le=1000, description="Slow EMA length")


class AwesomeOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Awesome Oscillator indicator."""
    fast_length: int = Field(5, ge=1, le=1000, description="Fast SMA length")
    slow_length: int = Field(34, ge=1, le=1000, description="Slow SMA length")


class PriceOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Price Oscillator indicator."""
    fast_length: int = Field(12, ge=1, le=1000, description="Fast MA length")
    slow_length: int = Field(26, ge=1, le=1000, description="Slow MA length")
    signal_length: int = Field(9, ge=1, le=1000, description="Signal line length")


class Detrended_PriceOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Detrended Price Oscillator indicator."""
    length: int = Field(20, ge=1, le=1000, description="DPO period")


class CoppockCurveArgs(BaseIndicatorArgs):
    """Arguments for Coppock Curve indicator."""
    wma_length: int = Field(10, ge=1, le=1000, description="WMA period")
    roc1_length: int = Field(14, ge=1, le=1000, description="First ROC period")
    roc2_length: int = Field(11, ge=1, le=1000, description="Second ROC period")


class ConnorsRSIArgs(BaseIndicatorArgs):
    """Arguments for Connors RSI indicator."""
    rsi_length: int = Field(3, ge=1, le=1000, description="RSI period")
    streak_length: int = Field(2, ge=1, le=1000, description="Streak period")
    rank_length: int = Field(100, ge=1, le=1000, description="Rank period")


class ChopZoneArgs(BaseIndicatorArgs):
    """Arguments for Chop Zone indicator."""
    ema_length: int = Field(34, ge=1, le=1000, description="EMA period")
    
    
class ChoppinessIndexArgs(BaseIndicatorArgs):
    """Arguments for Choppiness Index indicator."""
    length: int = Field(14, ge=1, le=1000, description="Choppiness period")


class ChandeKrollStopArgs(BaseIndicatorArgs):
    """Arguments for Chande Kroll Stop indicator."""
    length: int = Field(10, ge=1, le=1000, description="Period length")
    mult: float = Field(3.0, ge=0.1, le=10.0, description="Multiplier")


class ChandeMomentumOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Chande Momentum Oscillator indicator."""
    length: int = Field(20, ge=1, le=1000, description="CMO period")


class BalanceOfPowerArgs(BaseIndicatorArgs):
    """Arguments for Balance of Power indicator."""
    pass  # Balance of Power doesn't require additional parameters


class AccumulationDistributionArgs(BaseIndicatorArgs):
    """Arguments for Accumulation/Distribution indicator."""
    pass  # A/D doesn't require additional parameters


class AccumulativeSwingIndexArgs(BaseIndicatorArgs):
    """Arguments for Accumulative Swing Index indicator."""
    limit_move: float = Field(0.5, ge=0.1, le=100000.0, description="Limit move value")


class AdvanceDeclineArgs(BaseIndicatorArgs):
    """Arguments for Advance/Decline indicator."""
    length: int = Field(1, ge=1, le=2000, description="A/D period")


class EaseOfMovementArgs(BaseIndicatorArgs):
    """Arguments for Ease of Movement indicator."""
    length: int = Field(14, ge=1, le=1000, description="EOM period")


class ElderForceIndexArgs(BaseIndicatorArgs):
    """Arguments for Elder's Force Index indicator."""
    length: int = Field(13, ge=1, le=1000, description="Force Index period")


class EnvelopesArgs(BaseIndicatorArgs):
    """Arguments for Envelopes indicator."""
    length: int = Field(20, ge=1, le=1000, description="MA period")
    percent: float = Field(2.5, ge=0.1, le=50.0, description="Envelope percentage")


class FisherTransformArgs(BaseIndicatorArgs):
    """Arguments for Fisher Transform indicator."""
    length: int = Field(10, ge=1, le=1000, description="Fisher Transform period")


class HistoricalVolatilityArgs(BaseIndicatorArgs):
    """Arguments for Historical Volatility indicator."""
    length: int = Field(30, ge=1, le=1000, description="Volatility period")


class KlingerOscillatorArgs(BaseIndicatorArgs):
    """Arguments for Klinger Oscillator indicator."""
    fast_length: int = Field(34, ge=1, le=1000, description="Fast EMA length")
    slow_length: int = Field(55, ge=1, le=1000, description="Slow EMA length")
    signal_length: int = Field(13, ge=1, le=1000, description="Signal line length")


class KnowSureThingArgs(BaseIndicatorArgs):
    """Arguments for Know Sure Thing indicator."""
    roc1_length: int = Field(10, ge=1, le=1000, description="First ROC period")
    roc2_length: int = Field(15, ge=1, le=1000, description="Second ROC period")
    roc3_length: int = Field(20, ge=1, le=1000, description="Third ROC period")
    roc4_length: int = Field(30, ge=1, le=1000, description="Fourth ROC period")
    sma1_length: int = Field(10, ge=1, le=1000, description="First SMA period")
    sma2_length: int = Field(10, ge=1, le=1000, description="Second SMA period")
    sma3_length: int = Field(10, ge=1, le=1000, description="Third SMA period")
    sma4_length: int = Field(15, ge=1, le=1000, description="Fourth SMA period")
    signal_length: int = Field(9, ge=1, le=1000, description="Signal line length")


class LeastSquaresMAArgs(BaseIndicatorArgs):
    """Arguments for Least Squares Moving Average indicator."""
    length: int = Field(25, ge=1, le=1000, description="LSMA period")


class LinearRegressionSlopeArgs(BaseIndicatorArgs):
    """Arguments for Linear Regression Slope indicator."""
    length: int = Field(14, ge=1, le=1000, description="Linear regression period")


class MACrossArgs(BaseIndicatorArgs):
    """Arguments for MA Cross indicator."""
    fast_length: int = Field(10, ge=1, le=1000, description="Fast MA length")
    slow_length: int = Field(21, ge=1, le=1000, description="Slow MA length")


class MAWithEMACrossArgs(BaseIndicatorArgs):
    """Arguments for MA with EMA Cross indicator."""
    ma_length: int = Field(21, ge=1, le=1000, description="MA length")
    ema_length: int = Field(13, ge=1, le=1000, description="EMA length")


class MajorityRuleArgs(BaseIndicatorArgs):
    """Arguments for Majority Rule indicator."""
    length: int = Field(4, ge=1, le=100, description="Majority rule period")


class MassIndexArgs(BaseIndicatorArgs):
    """Arguments for Mass Index indicator."""
    length: int = Field(25, ge=1, le=1000, description="Mass Index period")


class McGinleyDynamicArgs(BaseIndicatorArgs):
    """Arguments for McGinley Dynamic indicator."""
    length: int = Field(14, ge=1, le=1000, description="McGinley Dynamic period")
    constant: float = Field(0.6, ge=0.1, le=2.0, description="Constant factor")


class MovingAverageChannelArgs(BaseIndicatorArgs):
    """Arguments for Moving Average Channel indicator."""
    length: int = Field(20, ge=1, le=1000, description="MA period")
    percent: float = Field(2.5, ge=0.1, le=50.0, description="Channel percentage")


class MovingAverageDoubleArgs(BaseIndicatorArgs):
    """Arguments for Moving Average Double indicator."""
    length: int = Field(14, ge=1, le=1000, description="MA period")


class MovingAverageHammingArgs(BaseIndicatorArgs):
    """Arguments for Moving Average Hamming indicator."""
    length: int = Field(14, ge=1, le=1000, description="Hamming MA period")


class MovingAverageMultipleArgs(BaseIndicatorArgs):
    """Arguments for Moving Average Multiple indicator."""
    length1: int = Field(10, ge=1, le=1000, description="First MA period")
    length2: int = Field(20, ge=1, le=1000, description="Second MA period")
    length3: int = Field(30, ge=1, le=1000, description="Third MA period")


class MovingAverageTripleArgs(BaseIndicatorArgs):
    """Arguments for Moving Average Triple indicator."""
    length: int = Field(14, ge=1, le=1000, description="MA period")


class NetVolumeArgs(BaseIndicatorArgs):
    """Arguments for Net Volume indicator."""
    pass  # Net Volume doesn't require additional parameters


class PriceVolumeTrendArgs(BaseIndicatorArgs):
    """Arguments for Price Volume Trend indicator."""
    pass  # PVT doesn't require additional parameters


class RelativeVigorIndexArgs(BaseIndicatorArgs):
    """Arguments for Relative Vigor Index indicator."""
    length: int = Field(10, ge=1, le=1000, description="RVI period")


class SMIErgodicArgs(BaseIndicatorArgs):
    """Arguments for SMI Ergodic indicator."""
    length1: int = Field(5, ge=1, le=1000, description="First smoothing period")
    length2: int = Field(20, ge=1, le=1000, description="Second smoothing period")
    signal_length: int = Field(5, ge=1, le=1000, description="Signal line period")


class DirectionalMovementArgs(BaseIndicatorArgs):
    """Arguments for Directional Movement indicator."""
    adx_smoothing: int = Field(14, ge=1, le=1000, description="ADX smoothing period")
    di_length: int = Field(14, ge=1, le=1000, description="DI length")


class ChaikinVolatilityArgs(BaseIndicatorArgs):
    """Arguments for Chaikin Volatility indicator."""
    ema_length: int = Field(10, ge=1, le=1000, description="EMA period")
    roc_length: int = Field(10, ge=1, le=1000, description="ROC period")


class ChartDrawingPoint(BaseModel):
    """Point for chart drawings."""
    x: str = Field(..., description="Date in YYYY-MM-DD format")
    y: float = Field(..., description="Price level")


class ChartDrawing(BaseModel):
    """Chart drawing/annotation configuration."""
    type: str = Field(..., description="Drawing type (e.g., trendline, rectangle)")
    points: List[ChartDrawingPoint] = Field(..., description="Drawing points")
    color: Optional[str] = Field(None, description="Drawing color")
    width: Optional[int] = Field(None, description="Line width")


class ChartConfig(BaseModel):
    """Chart configuration model."""
    symbol: str = Field(..., min_length=1, description="Trading symbol in format EXCHANGE:SYMBOL")
    interval: Literal["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1D", "1W", "1M"] = Field(
        ..., description="Chart time interval"
    )
    chartType: Literal[
        "candlestick", "line", "area", "bar", "heikin_ashi", 
        "hollow_candle", "baseline", "hi_lo", "column"
    ] = Field(..., description="Type of chart to generate")
    duration: Literal["1D", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"] = Field(
        default="1M", description="Chart duration/timeframe"
    )
    width: int = Field(default=800, ge=400, le=2000, description="Chart width in pixels")
    height: int = Field(default=600, ge=300, le=1500, description="Chart height in pixels")
    indicators: List[ChartIndicator] = Field(default_factory=list, description="Technical indicators")
    drawings: List[ChartDrawing] = Field(default_factory=list, description="Chart drawings and annotations")
    theme: Literal["light", "dark"] = Field(default="light", description="Chart theme")
    showVolume: bool = Field(default=True, description="Show volume indicator")
    showGrid: bool = Field(default=True, description="Show chart grid")
    timezone: str = Field(default="America/New_York", description="Chart timezone")


class User(BaseModel):
    """User model for authentication."""
    id: int
    username: str
    password: str


class ChartRequest(BaseModel):
    """Chart generation request model."""
    id: int
    requestId: str
    symbol: str
    interval: str
    chartType: str
    duration: str
    width: int
    height: int
    indicators: Optional[List[Dict[str, Any]]] = None
    drawings: Optional[List[Dict[str, Any]]] = None
    theme: str
    showVolume: bool
    showGrid: bool
    timezone: str
    status: str
    chartUrl: Optional[str] = None
    base64Data: Optional[str] = None
    errorMessage: Optional[str] = None
    processingTime: Optional[int] = None
    createdAt: datetime
    completedAt: Optional[datetime] = None


class SseEvent(BaseModel):
    """Server-Sent Events model."""
    id: int
    requestId: str
    eventType: str
    message: str
    timestamp: datetime


class ChartImgResponse(BaseModel):
    """Response from Chart-IMG API."""
    success: bool
    url: Optional[str] = None
    base64: Optional[str] = None
    error: Optional[str] = None
    processingTime: Optional[int] = None


class InsertUser(BaseModel):
    """User creation model."""
    username: str
    password: str


class InsertChartRequest(BaseModel):
    """Chart request creation model."""
    symbol: str
    interval: str
    chartType: str
    duration: str = "1M"
    width: int = 800
    height: int = 600
    indicators: Optional[List[Dict[str, Any]]] = None
    drawings: Optional[List[Dict[str, Any]]] = None
    theme: str = "light"
    showVolume: bool = True
    showGrid: bool = True
    timezone: str = "America/New_York"


class InsertSseEvent(BaseModel):
    """SSE event creation model."""
    requestId: str
    eventType: str
    message: str