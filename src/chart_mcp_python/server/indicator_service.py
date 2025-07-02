"""Indicator service for managing TradingView indicators."""

from typing import Any, Dict, List, Optional, Union
from ..shared.schema import BaseIndicatorArgs


class IndicatorService:
    """Service for handling TradingView indicator configurations."""

    def __init__(self) -> None:
        """Initialize the indicator service."""
        self.indicator_name_map = self._build_indicator_name_map()

    def _build_indicator_name_map(self) -> Dict[str, str]:
        """Build mapping from tool names to Chart-IMG indicator names."""
        return {
            # Moving Averages
            "add_sma": "Moving Average",
            "add_ema": "Moving Average Exponential", 
            "add_wma": "Moving Average Weighted",
            "add_hull_ma": "Hull Moving Average",
            "add_arnaud_legoux_ma": "Arnaud Legoux Moving Average",
            "add_double_ema": "Double EMA",
            "add_triple_ma": "Moving Average Triple",
            "add_adaptive_ma": "Moving Average Adaptive",
            "add_least_squares_ma": "Least Squares Moving Average",
            "add_mcginley_dynamic": "McGinley Dynamic",
            "add_ma_channel": "Moving Average Channel",
            "add_ma_multiple": "Moving Average Multiple",
            "add_ma_hamming": "Moving Average Hamming",
            
            # Oscillators
            "add_rsi": "Relative Strength Index",
            "add_macd": "MACD",
            "add_stochastic": "Stochastic",
            "add_cci": "Commodity Channel Index",
            "add_williams_r": "Williams %R",
            "add_momentum": "Momentum",
            "add_roc": "Rate Of Change", 
            "add_mfi": "Money Flow Index",
            "add_ultimate_oscillator": "Ultimate Oscillator",
            "add_awesome_oscillator": "Awesome Oscillator",
            "add_price_oscillator": "Price Oscillator",
            "add_detrended_price_oscillator": "Detrended Price Oscillator",
            "add_chaikin_oscillator": "Chaikin Oscillator",
            "add_klinger_oscillator": "Klinger Oscillator",
            "add_connors_rsi": "Connors RSI",
            "add_chande_momentum_oscillator": "Chande Momentum Oscillator",
            "add_relative_vigor_index": "Relative Vigor Index",
            "add_smi_ergodic": "SMI Ergodic",
            
            # Volume Indicators
            "add_volume": "Volume",
            "add_obv": "On Balance Volume",
            "add_cmf": "Chaikin Money Flow",
            "add_ease_of_movement": "Ease of Movement",
            "add_accumulation_distribution": "Accumulation/Distribution",
            "add_elder_force_index": "Elder's Force Index",
            "add_net_volume": "Net Volume",
            "add_price_volume_trend": "Price Volume Trend",
            
            # Volatility Indicators
            "add_bollinger_bands": "Bollinger Bands",
            "add_atr": "Average True Range",
            "add_keltner_channels": "Keltner Channels",
            "add_donchian_channels": "Donchian Channels",
            "add_historical_volatility": "Historical Volatility",
            "add_envelopes": "Envelopes",
            "add_chaikin_volatility": "Chaikin Volatility",
            
            # Trend Indicators
            "add_parabolic_sar": "Parabolic SAR",
            "add_adx": "Average Directional Index",
            "add_aroon": "Aroon",
            "add_ichimoku": "Ichimoku Cloud",
            "add_linear_regression": "Linear Regression Curve",
            "add_linear_regression_slope": "Linear Regression Slope",
            "add_directional_movement": "Directional Movement",
            "add_ma_cross": "MA Cross",
            "add_ma_with_ema_cross": "MA with EMA Cross",
            "add_coppock_curve": "Coppock Curve",
            
            # Other Indicators
            "add_balance_of_power": "Balance of Power",
            "add_accumulative_swing_index": "Accumulative Swing Index",
            "add_advance_decline": "Advance/Decline",
            "add_fisher_transform": "Fisher Transform",
            "add_know_sure_thing": "Know Sure Thing",
            "add_price_channel": "Price Channel",
            "add_chop_zone": "Chop Zone",
            "add_choppiness_index": "Choppiness Index",
            "add_chande_kroll_stop": "Chande Kroll Stop",
            "add_majority_rule": "Majority Rule",
            "add_mass_index": "Mass Index",
        }

    def build_indicator_config(
        self, 
        tool_name: str, 
        args: BaseIndicatorArgs
    ) -> Dict[str, Any]:
        """Build indicator configuration for Chart-IMG API."""
        indicator_name = self.indicator_name_map.get(tool_name)
        if not indicator_name:
            raise ValueError(f"Unknown indicator tool: {tool_name}")

        config: Dict[str, Any] = {
            "name": indicator_name,
            "inputs": [],
        }

        # Build inputs based on indicator type
        inputs = self._build_indicator_inputs(tool_name, args)
        if inputs:
            config["inputs"] = inputs

        # Build styles (overrides)
        styles = self._build_indicator_styles(args)
        if styles:
            config["styles"] = styles

        return config

    def _build_indicator_inputs(
        self, 
        tool_name: str, 
        args: BaseIndicatorArgs
    ) -> List[Union[int, float, str]]:
        """Build input parameters for specific indicators."""
        inputs = []
        
        # Simple length-based indicators
        if hasattr(args, 'length') and tool_name in [
            'add_sma', 'add_ema', 'add_wma', 'add_rsi', 'add_cci', 'add_williams_r',
            'add_momentum', 'add_roc', 'add_mfi', 'add_atr', 'add_aroon', 'add_cmf',
            'add_ease_of_movement', 'add_elder_force_index', 'add_historical_volatility',
            'add_linear_regression', 'add_linear_regression_slope', 'add_fisher_transform',
            'add_choppiness_index', 'add_chande_momentum_oscillator', 'add_mass_index',
            'add_least_squares_ma', 'add_mcginley_dynamic', 'add_relative_vigor_index',
            'add_detrended_price_oscillator', 'add_hull_ma', 'add_advance_decline'
        ]:
            inputs.append(args.length)
            
        # Hull MA specific
        elif tool_name == 'add_hull_ma' and hasattr(args, 'length'):
            inputs.append(args.length)
            
        # MACD
        elif tool_name == 'add_macd' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length, args.signal_length])
            
        # Bollinger Bands
        elif tool_name == 'add_bollinger_bands' and hasattr(args, 'length'):
            inputs.extend([args.length, args.mult])
            
        # Stochastic
        elif tool_name == 'add_stochastic' and hasattr(args, 'k_length'):
            inputs.extend([args.k_length, args.k_smoothing, args.d_length])
            
        # ADX
        elif tool_name == 'add_adx' and hasattr(args, 'smoothing'):
            inputs.extend([args.smoothing, args.di_length])
            
        # Arnaud Legoux MA
        elif tool_name == 'add_arnaud_legoux_ma' and hasattr(args, 'length'):
            inputs.extend([args.length, args.offset, args.sigma])
            
        # Keltner Channels
        elif tool_name == 'add_keltner_channels' and hasattr(args, 'length'):
            inputs.extend([args.length, args.mult, args.atr_length])
            
        # Donchian Channels
        elif tool_name == 'add_donchian_channels' and hasattr(args, 'length'):
            inputs.append(args.length)
            
        # Parabolic SAR
        elif tool_name == 'add_parabolic_sar' and hasattr(args, 'start'):
            inputs.extend([args.start, args.increment, args.maximum])
            
        # Ichimoku
        elif tool_name == 'add_ichimoku' and hasattr(args, 'conversion_line_length'):
            inputs.extend([
                args.conversion_line_length, 
                args.base_line_length,
                args.leading_span_b_length, 
                args.displacement
            ])
            
        # Ultimate Oscillator
        elif tool_name == 'add_ultimate_oscillator' and hasattr(args, 'length1'):
            inputs.extend([args.length1, args.length2, args.length3])
            
        # Awesome Oscillator
        elif tool_name == 'add_awesome_oscillator' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length])
            
        # Price Oscillator
        elif tool_name == 'add_price_oscillator' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length, args.signal_length])
            
        # Chaikin Oscillator
        elif tool_name == 'add_chaikin_oscillator' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length])
            
        # Klinger Oscillator
        elif tool_name == 'add_klinger_oscillator' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length, args.signal_length])
            
        # Connors RSI
        elif tool_name == 'add_connors_rsi' and hasattr(args, 'rsi_length'):
            inputs.extend([args.rsi_length, args.streak_length, args.rank_length])
            
        # Know Sure Thing
        elif tool_name == 'add_know_sure_thing' and hasattr(args, 'roc1_length'):
            inputs.extend([
                args.roc1_length, args.roc2_length, args.roc3_length, args.roc4_length,
                args.sma1_length, args.sma2_length, args.sma3_length, args.sma4_length,
                args.signal_length
            ])
            
        # MA Cross
        elif tool_name == 'add_ma_cross' and hasattr(args, 'fast_length'):
            inputs.extend([args.fast_length, args.slow_length])
            
        # MA with EMA Cross
        elif tool_name == 'add_ma_with_ema_cross' and hasattr(args, 'ma_length'):
            inputs.extend([args.ma_length, args.ema_length])
            
        # Coppock Curve
        elif tool_name == 'add_coppock_curve' and hasattr(args, 'wma_length'):
            inputs.extend([args.wma_length, args.roc1_length, args.roc2_length])
            
        # Envelopes
        elif tool_name == 'add_envelopes' and hasattr(args, 'length'):
            inputs.extend([args.length, args.percent])
            
        # Chaikin Volatility
        elif tool_name == 'add_chaikin_volatility' and hasattr(args, 'ema_length'):
            inputs.extend([args.ema_length, args.roc_length])
            
        # Directional Movement
        elif tool_name == 'add_directional_movement' and hasattr(args, 'adx_smoothing'):
            inputs.extend([args.adx_smoothing, args.di_length])
            
        # Moving Average Channel
        elif tool_name == 'add_ma_channel' and hasattr(args, 'length'):
            inputs.extend([args.length, args.percent])
            
        # Moving Average Multiple
        elif tool_name == 'add_ma_multiple' and hasattr(args, 'length1'):
            inputs.extend([args.length1, args.length2, args.length3])
            
        # SMI Ergodic
        elif tool_name == 'add_smi_ergodic' and hasattr(args, 'length1'):
            inputs.extend([args.length1, args.length2, args.signal_length])
            
        # McGinley Dynamic
        elif tool_name == 'add_mcginley_dynamic' and hasattr(args, 'length'):
            inputs.extend([args.length, args.constant])
            
        # Chande Kroll Stop
        elif tool_name == 'add_chande_kroll_stop' and hasattr(args, 'length'):
            inputs.extend([args.length, args.mult])
            
        # Accumulative Swing Index
        elif tool_name == 'add_accumulative_swing_index' and hasattr(args, 'limit_move'):
            inputs.append(args.limit_move)
            
        # Chop Zone
        elif tool_name == 'add_chop_zone' and hasattr(args, 'ema_length'):
            inputs.append(args.ema_length)
            
        # Majority Rule
        elif tool_name == 'add_majority_rule' and hasattr(args, 'length'):
            inputs.append(args.length)
            
        # Volume with MA
        elif tool_name == 'add_volume' and hasattr(args, 'show_ma') and args.show_ma:
            inputs.append(args.ma_length)

        return inputs

    def _build_indicator_styles(self, args: BaseIndicatorArgs) -> Dict[str, Any]:
        """Build style overrides for indicators."""
        styles = {}
        
        # Main plot style
        plot_style = {}
        if args.color:
            plot_style["color"] = args.color
        if args.linewidth:
            plot_style["linewidth"] = args.linewidth
        if args.plottype:
            plot_style["plottype"] = args.plottype
            
        if plot_style:
            styles["plot_0"] = plot_style

        return styles

    def get_supported_indicators(self) -> List[str]:
        """Get list of all supported indicator tool names."""
        return list(self.indicator_name_map.keys())

    def get_indicator_display_name(self, tool_name: str) -> Optional[str]:
        """Get the display name for an indicator tool."""
        return self.indicator_name_map.get(tool_name)


# Global instance
indicator_service = IndicatorService()