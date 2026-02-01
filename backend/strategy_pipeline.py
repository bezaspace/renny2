import json
import os
import random
import urllib.parse
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import talib
import yfinance as yf
from talib import abstract
from vectorbt import settings
from vectorbt.portfolio.base import Portfolio
from vectorbt.portfolio.enums import Direction, DirectionConflictMode
from vectorbt.utils.colors import adjust_opacity
from vectorbt.utils.config import merge_dicts

from google.adk.agents import BaseAgent, InvocationContext, LlmAgent, SequentialAgent
from google.adk.tools import ToolContext
from google.adk.utils.context_utils import Aclosing
from google.adk.utils.instructions_utils import inject_session_state
from google.genai import types
from mcp import ClientSession, types as mcp_types
from mcp.client.streamable_http import streamablehttp_client

# --- Defaults aligned to inspiration.md ---
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"]
PLOT_TYPES = ["OHLC", "Candlestick"]
PATTERNS = talib.get_function_groups()["Pattern Recognition"]
STATS_TABLE_COLUMNS = ["Metric", "Buy & Hold", "Random (Median)", "Strategy", "Z-Score"]
DIRECTIONS = Direction._fields
CONFLICT_MODES = DirectionConflictMode._fields

DEFAULT_SYMBOL = "BTC-USD"
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_PLOT_TYPE = "OHLC"
DEFAULT_YF_AUTO_ADJUST = True
DEFAULT_YF_BACK_ADJUST = False
DEFAULT_ENTRY_PATTERNS = [
    "CDLHAMMER",
    "CDLINVERTEDHAMMER",
    "CDLPIERCING",
    "CDLMORNINGSTAR",
    "CDL3WHITESOLDIERS",
]
DEFAULT_EXIT_PATTERNS = [
    "CDLHANGINGMAN",
    "CDLSHOOTINGSTAR",
    "CDLEVENINGSTAR",
    "CDL3BLACKCROWS",
    "CDLDARKCLOUDCOVER",
]
DEFAULT_FEES = 0.1
DEFAULT_FIXED_FEES = 0.0
DEFAULT_SLIPPAGE = 5.0
DEFAULT_DIRECTION = DIRECTIONS[0]
DEFAULT_CONFLICT_MODE = CONFLICT_MODES[0]
DEFAULT_SIM_OPTIONS = ["allow_accumulate"]
DEFAULT_N_RANDOM_STRAT = 50
DEFAULT_PROB_OPTIONS = ["mimic_strategy"]
DEFAULT_ENTRY_PROB = 0.1
DEFAULT_EXIT_PROB = 0.1
DEFAULT_STATS_OPTIONS = ["incl_open"]
DEFAULT_SUBPLOTS = ["orders", "trade_pnl", "cum_returns"]
DEFAULT_METRIC = "Total Return [%]"

EXA_DEFAULT_TOOLS = [
    "web_search_exa",
    "company_research_exa",
    "deep_search_exa",
    "crawling_exa",
]

# --- Plot styling ---
COLOR_SCHEMA = settings["plotting"]["color_schema"]
BGCOLOR = "#ffffff"
DARK_BGCOLOR = "#f6f7fb"
FONTCOLOR = "#0f172a"
DARK_FONTCOLOR = "#475569"
GRIDCOLOR = "#e5e7eb"
ACTIVE_COLOR = "#2563EB"

DEFAULT_LAYOUT = dict(
    template="plotly_white",
    autosize=True,
    margin=dict(b=40, t=20),
    font=dict(color=FONTCOLOR),
    plot_bgcolor=BGCOLOR,
    paper_bgcolor=BGCOLOR,
    legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# --- Helpers ---

def _to_png(fig: go.Figure, width: int = 1200, height: int = 700) -> bytes:
    return pio.to_image(fig, format="png", width=width, height=height, scale=2)


def _load_df(state: Dict[str, Any]) -> pd.DataFrame:
    df_json = state.get("ohlcv_json")
    if not df_json:
        raise ValueError("No OHLCV data found in session state. Run data fetch first.")
    return pd.read_json(StringIO(df_json), orient="split")


def _direction_from_name(name: str) -> Direction:
    if name in Direction._fields:
        return getattr(Direction, name)
    return getattr(Direction, DEFAULT_DIRECTION)


def _conflict_from_name(name: str) -> DirectionConflictMode:
    if name in DirectionConflictMode._fields:
        return getattr(DirectionConflictMode, name)
    return getattr(DirectionConflictMode, DEFAULT_CONFLICT_MODE)


def _set_end_invocation(tool_context: ToolContext) -> None:
    # Use internal access to end the current invocation after a tool result.
    tool_context._invocation_context.end_invocation = True


def _exa_mcp_url() -> str:
    base_url = os.getenv("EXA_MCP_URL", "https://mcp.exa.ai/mcp").strip()
    tools_csv = os.getenv("EXA_MCP_TOOLS", ",".join(EXA_DEFAULT_TOOLS)).strip()
    if not tools_csv:
        return base_url
    url = urllib.parse.urlsplit(base_url)
    query = urllib.parse.parse_qs(url.query)
    query.setdefault("tools", [tools_csv])
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunsplit((url.scheme, url.netloc, url.path, new_query, url.fragment))


def _exa_allowed_tools() -> List[str]:
    tools_csv = os.getenv("EXA_MCP_TOOLS", ",".join(EXA_DEFAULT_TOOLS)).strip()
    if not tools_csv:
        return EXA_DEFAULT_TOOLS
    return [tool.strip() for tool in tools_csv.split(",") if tool.strip()]


def _exa_result_summary(result: mcp_types.CallToolResult) -> str:
    text_chunks: List[str] = []
    for content in result.content:
        if isinstance(content, mcp_types.TextContent):
            text = (content.text or "").strip()
            if text:
                text_chunks.append(text)
    if text_chunks:
        return "\n\n".join(text_chunks)
    if result.structuredContent is not None:
        try:
            return json.dumps(result.structuredContent, ensure_ascii=True, indent=2)
        except TypeError:
            return str(result.structuredContent)
    return "Exa MCP tool completed."


def _default_candle_settings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SettingType": [
                "BodyLong",
                "BodyVeryLong",
                "BodyShort",
                "BodyDoji",
                "ShadowLong",
                "ShadowVeryLong",
                "ShadowShort",
                "ShadowVeryShort",
                "Near",
                "Far",
                "Equal",
            ],
            "RangeType": [
                "RealBody",
                "RealBody",
                "RealBody",
                "HighLow",
                "RealBody",
                "RealBody",
                "Shadows",
                "HighLow",
                "HighLow",
                "HighLow",
                "HighLow",
            ],
            "AvgPeriod": [10, 10, 10, 10, 0, 0, 10, 10, 5, 5, 5],
            "Factor": [1.0, 3.0, 1.0, 0.1, 1.0, 2.0, 1.0, 0.1, 0.2, 0.6, 0.05],
        }
    )


def _apply_candle_settings(settings_rows: List[Dict[str, Any]]) -> None:
    from talib._ta_lib import CandleSettingType, RangeType, _ta_set_candle_settings

    for row in settings_rows:
        avg_period = row["AvgPeriod"]
        if isinstance(avg_period, float) and float.is_integer(avg_period):
            avg_period = int(avg_period)
        factor = float(row["Factor"])
        _ta_set_candle_settings(
            getattr(CandleSettingType, row["SettingType"]),
            getattr(RangeType, row["RangeType"]),
            avg_period,
            factor,
        )


# --- Tools ---

async def exa_mcp_research(
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Call Exa MCP tools for web research (auto-call allowed)."""
    allowed_tools = set(_exa_allowed_tools())
    if tool_name not in allowed_tools:
        raise ValueError(
            f"Unsupported Exa tool '{tool_name}'. Allowed: {', '.join(sorted(allowed_tools))}."
        )

    url = _exa_mcp_url()
    payload = arguments or {}

    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, payload)

    summary = _exa_result_summary(result)
    structured_payload = None
    if result.structuredContent is not None:
        try:
            structured_payload = json.loads(
                json.dumps(result.structuredContent, ensure_ascii=True)
            )
        except TypeError:
            structured_payload = str(result.structuredContent)
    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "exa_research",
        "summary": summary,
        "artifacts": [],
        "data": {
            "tool_name": tool_name,
            "arguments": payload,
            "structured": structured_payload,
        },
    }

async def reset_pipeline(tool_context: ToolContext) -> Dict[str, Any]:
    """Reset all stored pipeline state to start over."""
    keys = [
        "ohlcv_json",
        "ohlcv_index",
        "symbol",
        "period",
        "interval",
        "plot_type",
        "auto_adjust",
        "back_adjust",
        "entry_patterns",
        "exit_patterns",
        "entry_dates",
        "exit_dates",
        "candle_settings",
        "fees",
        "fixed_fees",
        "slippage",
        "direction",
        "conflict_mode",
        "sim_options",
        "n_random_strat",
        "prob_options",
        "entry_prob",
        "exit_prob",
        "stats_options",
        "subplots",
        "stats_json",
        "metric",
        "data_done",
        "signals_done",
        "backtest_done",
        "metric_done",
    ]
    for key in keys:
        if key in tool_context.state:
            tool_context.state[key] = None
    tool_context.state["data_done"] = False
    tool_context.state["signals_done"] = False
    tool_context.state["backtest_done"] = False
    tool_context.state["metric_done"] = False
    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "reset",
        "summary": "Pipeline state reset. Tell me the symbol, period, and interval to fetch data. (Informational only.)",
        "artifacts": [],
    }


async def fetch_ohlcv(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    plot_type: str = DEFAULT_PLOT_TYPE,
    auto_adjust: bool = DEFAULT_YF_AUTO_ADJUST,
    back_adjust: bool = DEFAULT_YF_BACK_ADJUST,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Fetch OHLCV data and render the base price/volume chart."""
    df = yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        actions=False,
        auto_adjust=auto_adjust,
        back_adjust=back_adjust,
    )
    if df is None or df.empty:
        raise ValueError("Empty data from yfinance.")

    tool_context.state["ohlcv_json"] = df.to_json(date_format="iso", orient="split")
    tool_context.state["ohlcv_index"] = pd.to_datetime(df.index).astype(str).tolist()
    tool_context.state["symbol"] = symbol
    tool_context.state["period"] = period
    tool_context.state["interval"] = interval
    tool_context.state["plot_type"] = plot_type
    tool_context.state["auto_adjust"] = auto_adjust
    tool_context.state["back_adjust"] = back_adjust
    tool_context.state["data_done"] = True
    tool_context.state["signals_done"] = False
    tool_context.state["backtest_done"] = False
    tool_context.state["metric_done"] = False

    fig = df.vbt.ohlcv.plot(
        plot_type=plot_type,
        **merge_dicts(
            DEFAULT_LAYOUT,
            dict(
                width=None,
                height=700,
                margin=dict(r=40),
                hovermode="closest",
                xaxis2=dict(title="Date"),
                yaxis2=dict(title="Volume"),
                yaxis=dict(title="Price"),
            ),
        ),
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR)
    fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)

    png = _to_png(fig, width=1200, height=700)
    filename = "ohlcv_base.png"
    await tool_context.save_artifact(
        filename=filename,
        artifact=types.Part.from_bytes(data=png, mime_type="image/png"),
    )

    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "data",
        "summary": (
            f"Fetched {symbol} data (period {period}, interval {interval}). "
            "Confirm if you want to generate candlestick pattern signals next. "
            "(Informational only.)"
        ),
        "artifacts": [
            {
                "filename": filename,
                "label": "OHLCV (base)",
                "mime_type": "image/png",
            }
        ],
    }


async def build_signals(
    entry_patterns: Optional[List[str]] = None,
    exit_patterns: Optional[List[str]] = None,
    entry_dates: Optional[List[str]] = None,
    exit_dates: Optional[List[str]] = None,
    candle_settings: Optional[List[Dict[str, Any]]] = None,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Generate candlestick pattern signals and render price chart with markers."""
    df = _load_df(tool_context.state)

    if candle_settings:
        _apply_candle_settings(candle_settings)
    else:
        candle_settings = _default_candle_settings().to_dict("records")

    entry_patterns = [p for p in (entry_patterns or DEFAULT_ENTRY_PATTERNS) if p != "CUSTOM"]
    exit_patterns = [p for p in (exit_patterns or DEFAULT_EXIT_PATTERNS) if p != "CUSTOM"]
    entry_dates = entry_dates or []
    exit_dates = exit_dates or []

    talib_inputs = {
        "open": df["Open"].values,
        "high": df["High"].values,
        "low": df["Low"].values,
        "close": df["Close"].values,
        "volume": df["Volume"].values,
    }

    entry_patterns_with_custom = list(entry_patterns) + ["CUSTOM"]
    exit_patterns_with_custom = list(exit_patterns) + ["CUSTOM"]
    all_patterns = list(set(entry_patterns_with_custom + exit_patterns_with_custom))

    signal_df = pd.DataFrame.vbt.empty(
        (len(df.index), len(all_patterns)),
        fill_value=0.0,
        index=df.index,
        columns=all_patterns,
    )

    for pattern in all_patterns:
        if pattern != "CUSTOM":
            signal_df[pattern] = abstract.Function(pattern)(talib_inputs)

    signal_df.loc[entry_dates, "CUSTOM"] += 100.0
    signal_df.loc[exit_dates, "CUSTOM"] += -100.0

    entry_signal_df = signal_df[entry_patterns_with_custom]
    exit_signal_df = signal_df[exit_patterns_with_custom]

    entry_df = entry_signal_df[(entry_signal_df > 0).any(axis=1)]
    entry_labels = []
    for _, row in entry_df.iterrows():
        entry_labels.append("<br>".join(row.index[row != 0]))
    entry_labels = np.asarray(entry_labels)

    exit_df = exit_signal_df[(exit_signal_df < 0).any(axis=1)]
    exit_labels = []
    for _, row in exit_df.iterrows():
        exit_labels.append("<br>".join(row.index[row != 0]))
    exit_labels = np.asarray(exit_labels)

    highest_high = df["High"].max()
    lowest_low = df["Low"].min()
    distance = (highest_high - lowest_low) / 5
    entry_y = df.loc[entry_df.index, "Low"] - distance
    entry_y.index = pd.to_datetime(entry_y.index)
    exit_y = df.loc[exit_df.index, "High"] + distance
    exit_y.index = pd.to_datetime(exit_y.index)

    entry_signals = pd.Series.vbt.empty_like(entry_y, True)
    exit_signals = pd.Series.vbt.empty_like(exit_y, True)

    plot_type = tool_context.state.get("plot_type") or DEFAULT_PLOT_TYPE
    fig = df.vbt.ohlcv.plot(
        plot_type=plot_type,
        **merge_dicts(
            DEFAULT_LAYOUT,
            dict(
                width=None,
                height=700,
                margin=dict(r=40),
                hovermode="closest",
                xaxis2=dict(title="Date"),
                yaxis2=dict(title="Volume"),
                yaxis=dict(title="Price"),
            ),
        ),
    )
    entry_signals.vbt.signals.plot_as_entry_markers(
        y=entry_y,
        trace_kwargs=dict(
            customdata=entry_labels[:, None],
            hovertemplate="%{x}<br>%{customdata[0]}",
            name="Bullish signal",
        ),
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig,
    )
    exit_signals.vbt.signals.plot_as_exit_markers(
        y=exit_y,
        trace_kwargs=dict(
            customdata=exit_labels[:, None],
            hovertemplate="%{x}<br>%{customdata[0]}",
            name="Bearish signal",
        ),
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig,
    )
    fig.update_xaxes(gridcolor=GRIDCOLOR)
    fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)

    png = _to_png(fig, width=1200, height=700)
    filename = "ohlcv_signals.png"
    await tool_context.save_artifact(
        filename=filename,
        artifact=types.Part.from_bytes(data=png, mime_type="image/png"),
    )

    tool_context.state["entry_patterns"] = entry_patterns
    tool_context.state["exit_patterns"] = exit_patterns
    tool_context.state["entry_dates"] = entry_dates
    tool_context.state["exit_dates"] = exit_dates
    tool_context.state["candle_settings"] = candle_settings
    tool_context.state["signals_done"] = True
    tool_context.state["backtest_done"] = False
    tool_context.state["metric_done"] = False

    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "signals",
        "summary": "Signals generated. Confirm if you want to run the backtest next. (Informational only.)",
        "artifacts": [
            {
                "filename": filename,
                "label": "OHLCV + signals",
                "mime_type": "image/png",
            }
        ],
    }


def _simulate_portfolio(
    df: pd.DataFrame,
    interval: str,
    entry_patterns: List[str],
    exit_patterns: List[str],
    entry_dates: List[str],
    exit_dates: List[str],
    fees: float,
    fixed_fees: float,
    slippage: float,
    direction: Direction,
    conflict_mode: DirectionConflictMode,
    sim_options: List[str],
    n_random_strat: int,
    prob_options: List[str],
    entry_prob: float,
    exit_prob: float,
):
    talib_inputs = {
        "open": df["Open"].values,
        "high": df["High"].values,
        "low": df["Low"].values,
        "close": df["Close"].values,
        "volume": df["Volume"].values,
    }
    entry_patterns = [p for p in entry_patterns if p != "CUSTOM"] + ["CUSTOM"]
    exit_patterns = [p for p in exit_patterns if p != "CUSTOM"] + ["CUSTOM"]
    all_patterns = list(set(entry_patterns + exit_patterns))
    entry_i = [all_patterns.index(p) for p in entry_patterns]
    exit_i = [all_patterns.index(p) for p in exit_patterns]
    signals = np.full((len(df.index), len(all_patterns)), 0.0, dtype=np.float64)
    for i, pattern in enumerate(all_patterns):
        if pattern != "CUSTOM":
            signals[:, i] = abstract.Function(pattern)(talib_inputs)
    signals[np.flatnonzero(df.index.isin(entry_dates)), all_patterns.index("CUSTOM")] += 100.0
    signals[np.flatnonzero(df.index.isin(exit_dates)), all_patterns.index("CUSTOM")] += -100.0
    signals /= 100.0

    def _generate_size(local_signals: np.ndarray) -> np.ndarray:
        entry_signals = local_signals[:, entry_i]
        exit_signals = local_signals[:, exit_i]
        return np.where(entry_signals > 0, entry_signals, 0).sum(axis=1) + np.where(
            exit_signals < 0, exit_signals, 0
        ).sum(axis=1)

    main_size = np.empty((len(df.index),), dtype=np.float64)
    main_size[0] = 0
    main_size[1:] = _generate_size(signals)[:-1]

    hold_size = np.full_like(main_size, 0.0)
    hold_size[0] = np.inf

    def _shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    rand_size = np.empty((len(df.index), n_random_strat), dtype=np.float64)
    rand_size[0] = 0
    if "mimic_strategy" in prob_options:
        for i in range(n_random_strat):
            rand_signals = _shuffle_along_axis(signals, 0)
            rand_size[1:, i] = _generate_size(rand_signals)[:-1]
    else:
        entry_signals = pd.DataFrame.vbt.signals.generate_random(
            (rand_size.shape[0] - 1, rand_size.shape[1]), prob=entry_prob / 100
        ).values
        exit_signals = pd.DataFrame.vbt.signals.generate_random(
            (rand_size.shape[0] - 1, rand_size.shape[1]), prob=exit_prob / 100
        ).values
        rand_size[1:, :] = np.where(entry_signals, 1.0, 0.0) - np.where(exit_signals, 1.0, 0.0)

    def _simulate(size, init_cash="autoalign"):
        return Portfolio.from_signals(
            close=df["Close"],
            entries=size > 0,
            exits=size < 0,
            price=df["Open"],
            size=np.abs(size),
            direction=direction,
            upon_dir_conflict=conflict_mode,
            accumulate="allow_accumulate" in sim_options,
            init_cash=init_cash,
            fees=float(fees) / 100,
            fixed_fees=float(fixed_fees),
            slippage=(float(slippage) / 100) * (df["High"] / df["Open"] - 1),
            freq=interval,
        )

    aligned_portfolio = _simulate(np.hstack((main_size[:, None], rand_size)))
    aligned_portfolio = aligned_portfolio.replace(init_cash=aligned_portfolio.init_cash)
    main_portfolio = aligned_portfolio.iloc[0]
    rand_portfolio = aligned_portfolio.iloc[1:]
    hold_portfolio = _simulate(hold_size, init_cash=main_portfolio.init_cash)

    return main_portfolio, hold_portfolio, rand_portfolio


async def run_backtest(
    fees: float = DEFAULT_FEES,
    fixed_fees: float = DEFAULT_FIXED_FEES,
    slippage: float = DEFAULT_SLIPPAGE,
    direction: str = DEFAULT_DIRECTION,
    conflict_mode: str = DEFAULT_CONFLICT_MODE,
    sim_options: Optional[List[str]] = None,
    n_random_strat: int = DEFAULT_N_RANDOM_STRAT,
    prob_options: Optional[List[str]] = None,
    entry_prob: float = DEFAULT_ENTRY_PROB,
    exit_prob: float = DEFAULT_EXIT_PROB,
    stats_options: Optional[List[str]] = None,
    subplots: Optional[List[str]] = None,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Run the portfolio simulation and render performance charts."""
    df = _load_df(tool_context.state)

    entry_patterns = tool_context.state.get("entry_patterns") or DEFAULT_ENTRY_PATTERNS
    exit_patterns = tool_context.state.get("exit_patterns") or DEFAULT_EXIT_PATTERNS
    entry_dates = tool_context.state.get("entry_dates") or []
    exit_dates = tool_context.state.get("exit_dates") or []

    sim_options = sim_options or DEFAULT_SIM_OPTIONS
    prob_options = prob_options or DEFAULT_PROB_OPTIONS
    stats_options = stats_options or DEFAULT_STATS_OPTIONS
    subplots = subplots or DEFAULT_SUBPLOTS

    main_portfolio, hold_portfolio, rand_portfolio = _simulate_portfolio(
        df,
        tool_context.state.get("interval") or DEFAULT_INTERVAL,
        entry_patterns,
        exit_patterns,
        entry_dates,
        exit_dates,
        fees,
        fixed_fees,
        slippage,
        _direction_from_name(direction),
        _conflict_from_name(conflict_mode),
        sim_options,
        n_random_strat,
        prob_options,
        entry_prob,
        exit_prob,
    )

    subplot_settings = {}
    symbol = tool_context.state.get("symbol") or DEFAULT_SYMBOL
    if "cum_returns" in subplots:
        subplot_settings["cum_returns"] = dict(
            benchmark_kwargs=dict(
                trace_kwargs=dict(line=dict(color=adjust_opacity(COLOR_SCHEMA["yellow"], 0.5)), name=symbol)
            )
        )

    fig = main_portfolio.plot(
        subplots=subplots,
        subplot_settings=subplot_settings,
        **merge_dicts(
            DEFAULT_LAYOUT,
            dict(width=None, height=max(350, len(subplots) * 300)),
        ),
    )
    fig.update_traces(xaxis="x" if len(subplots) == 1 else "x" + str(len(subplots)))
    fig.update_xaxes(gridcolor=GRIDCOLOR)
    fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)

    def _chop_microseconds(delta):
        return delta - pd.Timedelta(microseconds=delta.microseconds, nanoseconds=delta.nanoseconds)

    def _metric_to_str(x):
        if isinstance(x, float):
            return "%.2f" % x
        if isinstance(x, pd.Timedelta):
            return str(_chop_microseconds(x))
        return str(x)

    incl_open = "incl_open" in stats_options
    use_positions = "use_positions" in stats_options
    main_stats = main_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions))
    hold_stats = hold_portfolio.stats(settings=dict(incl_open=True, use_positions=use_positions))
    rand_stats = rand_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions), agg_func=None)
    rand_stats_median = rand_stats.iloc[:, 3:].median(axis=0)
    rand_stats_mean = rand_stats.iloc[:, 3:].mean(axis=0)
    rand_stats_std = rand_stats.iloc[:, 3:].std(axis=0, ddof=0)
    stats_mean_diff = main_stats.iloc[3:] - rand_stats_mean

    def _to_float(x):
        if pd.isnull(x):
            return np.nan
        if isinstance(x, float):
            if np.allclose(x, 0):
                return 0.0
        if isinstance(x, pd.Timedelta):
            return float(x.total_seconds())
        return float(x)

    z = stats_mean_diff.apply(_to_float) / rand_stats_std.apply(_to_float)

    table_data = pd.DataFrame(columns=STATS_TABLE_COLUMNS)
    table_data.iloc[:, 0] = main_stats.index
    table_data.iloc[:, 1] = hold_stats.apply(_metric_to_str).values
    table_data.iloc[:3, 2] = table_data.iloc[:3, 1]
    table_data.iloc[3:, 2] = rand_stats_median.apply(_metric_to_str).values
    table_data.iloc[:, 3] = main_stats.apply(_metric_to_str).values
    table_data.iloc[3:, 4] = z.apply(_metric_to_str).values

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=STATS_TABLE_COLUMNS, fill_color=BGCOLOR, align="left"),
                cells=dict(values=[table_data[c] for c in STATS_TABLE_COLUMNS], align="left"),
            )
        ]
    )
    table_fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=400)

    portfolio_png = _to_png(fig, width=1200, height=800)
    stats_png = _to_png(table_fig, width=1200, height=400)

    portfolio_file = "portfolio.png"
    stats_file = "stats_table.png"
    await tool_context.save_artifact(
        filename=portfolio_file,
        artifact=types.Part.from_bytes(data=portfolio_png, mime_type="image/png"),
    )
    await tool_context.save_artifact(
        filename=stats_file,
        artifact=types.Part.from_bytes(data=stats_png, mime_type="image/png"),
    )

    tool_context.state["fees"] = fees
    tool_context.state["fixed_fees"] = fixed_fees
    tool_context.state["slippage"] = slippage
    tool_context.state["direction"] = direction
    tool_context.state["conflict_mode"] = conflict_mode
    tool_context.state["sim_options"] = sim_options
    tool_context.state["n_random_strat"] = n_random_strat
    tool_context.state["prob_options"] = prob_options
    tool_context.state["entry_prob"] = entry_prob
    tool_context.state["exit_prob"] = exit_prob
    tool_context.state["stats_options"] = stats_options
    tool_context.state["subplots"] = subplots

    stats_payload = {
        "main": {m: [_to_float(main_stats[m])] for m in main_stats.index[3:]},
        "hold": {m: [_to_float(hold_stats[m])] for m in main_stats.index[3:]},
        "rand": {m: rand_stats[m].apply(_to_float).values.tolist() for m in main_stats.index[3:]},
    }
    tool_context.state["stats_json"] = json.dumps(stats_payload)
    tool_context.state["backtest_done"] = True
    tool_context.state["metric_done"] = False

    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "backtest",
        "summary": "Backtest complete. Confirm which metric you want to visualize next. (Informational only.)",
        "artifacts": [
            {
                "filename": portfolio_file,
                "label": "Portfolio",
                "mime_type": "image/png",
            },
            {
                "filename": stats_file,
                "label": "Stats table",
                "mime_type": "image/png",
            },
        ],
        "metrics": list(stats_payload["main"].keys()),
    }


async def metric_distribution(
    metric: str = DEFAULT_METRIC,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Render the metric distribution plot across random, hold, and strategy."""
    stats_json = tool_context.state.get("stats_json")
    if not stats_json:
        raise ValueError("No stats found. Run backtest first.")
    stats_dict = json.loads(stats_json)
    if metric not in stats_dict["main"]:
        metric = DEFAULT_METRIC

    fig = go.Figure(
        data=[
            go.Box(
                x=stats_dict["rand"][metric],
                quartilemethod="linear",
                jitter=0.3,
                pointpos=1.8,
                boxpoints="all",
                boxmean="sd",
                hoveron="points",
                hovertemplate="%{x}<br>Random",
                name="",
                marker=dict(color=COLOR_SCHEMA["blue"], opacity=0.5, size=8),
            ),
            go.Box(
                x=stats_dict["hold"][metric],
                quartilemethod="linear",
                boxpoints="all",
                jitter=0,
                pointpos=1.8,
                hoveron="points",
                hovertemplate="%{x}<br>Buy & Hold",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)"),
                name="",
                marker=dict(color=COLOR_SCHEMA["orange"], size=8),
            ),
            go.Box(
                x=stats_dict["main"][metric],
                quartilemethod="linear",
                boxpoints="all",
                jitter=0,
                pointpos=1.8,
                hoveron="points",
                hovertemplate="%{x}<br>Strategy",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)"),
                name="",
                marker=dict(color=COLOR_SCHEMA["green"], size=8),
            ),
        ]
    )
    fig.update_layout(
        **merge_dicts(
            DEFAULT_LAYOUT,
            dict(
                height=400,
                showlegend=False,
                margin=dict(l=60, r=20, t=40, b=20),
                hovermode="closest",
                xaxis=dict(gridcolor=GRIDCOLOR, title=metric, side="top"),
                yaxis=dict(gridcolor=GRIDCOLOR),
            ),
        )
    )

    png = _to_png(fig, width=900, height=400)
    filename = "metric_distribution.png"
    await tool_context.save_artifact(
        filename=filename,
        artifact=types.Part.from_bytes(data=png, mime_type="image/png"),
    )

    tool_context.state["metric"] = metric
    tool_context.state["metric_done"] = True

    tool_context.actions.skip_summarization = True
    _set_end_invocation(tool_context)
    return {
        "step": "metric",
        "summary": f"Metric distribution for {metric} rendered. (Informational only.)",
        "artifacts": [
            {
                "filename": filename,
                "label": f"Metric distribution: {metric}",
                "mime_type": "image/png",
            }
        ],
    }


# --- Conditional Agent Wrapper ---

class ConditionalAgent(BaseAgent):
    wrapped_agent: BaseAgent
    should_run: Callable[[InvocationContext], bool]

    async def _run_async_impl(self, ctx: InvocationContext):
        if not self.should_run(ctx):
            return
        async with Aclosing(self.wrapped_agent.run_async(ctx)) as agen:
            async for event in agen:
                yield event


# --- Instructions ---

COMMON_RULES = (
    "You are Trading Copilot, an AI assistant for market analysis. "
    "Provide concise, practical explanations, clarify assumptions, and ask brief follow-up questions when details are missing. "
    "Do not provide personalized financial advice. "
    "Always include a short disclaimer that responses are informational only. "
    "At each step, ask for explicit confirmation before calling a tool, "
    "except you may auto-call exa_mcp_research when the user explicitly asks for research. "
    "For exa_mcp_research, set tool_name to one of: "
    + ", ".join(EXA_DEFAULT_TOOLS)
    + ". "
    "If the user is unsure, propose defaults, explain briefly, and ask to confirm. "
    "If the user says 'you decide', choose sensible defaults and ask to confirm."
)


async def _data_instruction(ctx) -> str:
    template = (
        COMMON_RULES
        + "\nCurrent data state: symbol={symbol?}, period={period?}, interval={interval?}, plot_type={plot_type?}. "
        + "If data not fetched, ask for: symbol, period (" + ", ".join(PERIODS) + "), interval ("
        + ", ".join(INTERVALS)
        + "), plot_type (OHLC or Candlestick), and whether to auto_adjust/back_adjust. "
        + "Defaults: symbol BTC-USD, period 1y, interval 1d, plot_type OHLC, auto_adjust true, back_adjust false. "
        + "After confirmation, call fetch_ohlcv."
    )
    return await inject_session_state(template, ctx)


async def _signals_instruction(ctx) -> str:
    template = (
        COMMON_RULES
        + "\nData fetched: {data_done?}. Signal state: entry_patterns={entry_patterns?}, exit_patterns={exit_patterns?}. "
        + "Ask for entry patterns, exit patterns (or say 'same as entry'), optional custom entry/exit dates, "
        + "and candle settings if the user wants to override defaults. "
        + "Defaults entry patterns: "
        + ", ".join(DEFAULT_ENTRY_PATTERNS)
        + ". Defaults exit patterns: "
        + ", ".join(DEFAULT_EXIT_PATTERNS)
        + ". After confirmation, call build_signals."
    )
    return await inject_session_state(template, ctx)


async def _backtest_instruction(ctx) -> str:
    template = (
        COMMON_RULES
        + "\nSignals generated: {signals_done?}. Backtest state: fees={fees?}, slippage={slippage?}, direction={direction?}. "
        + "Ask for fees (%), fixed_fees, slippage (% of H-O), direction (" + ", ".join(DIRECTIONS) + "), "
        + "conflict mode (" + ", ".join(CONFLICT_MODES) + "), number of random strategies, "
        + "and whether to mimic strategy or use entry/exit probabilities. "
        + "Defaults: fees 0.1, fixed_fees 0, slippage 5, direction " + DEFAULT_DIRECTION + ", conflict "
        + DEFAULT_CONFLICT_MODE
        + ", random_strategies 50, mimic_strategy on. After confirmation, call run_backtest."
    )
    return await inject_session_state(template, ctx)


async def _metric_instruction(ctx) -> str:
    template = (
        COMMON_RULES
        + "\nBacktest complete: {backtest_done?}. Current metric: {metric?}. "
        + "Ask which metric to visualize, default to 'Total Return [%]'. After confirmation, call metric_distribution."
    )
    return await inject_session_state(template, ctx)


# --- Build Pipeline ---

def build_agent_pipeline(model) -> BaseAgent:
    data_agent = LlmAgent(
        name="data_agent",
        model=model,
        description="Collects data parameters and fetches OHLCV.",
        instruction=_data_instruction,
        tools=[fetch_ohlcv, reset_pipeline, exa_mcp_research],
    )
    signals_agent = LlmAgent(
        name="signals_agent",
        model=model,
        description="Collects signal parameters and generates pattern signals.",
        instruction=_signals_instruction,
        tools=[build_signals, reset_pipeline, exa_mcp_research],
    )
    backtest_agent = LlmAgent(
        name="backtest_agent",
        model=model,
        description="Collects backtest parameters and runs the simulation.",
        instruction=_backtest_instruction,
        tools=[run_backtest, reset_pipeline, exa_mcp_research],
    )
    metric_agent = LlmAgent(
        name="metric_agent",
        model=model,
        description="Collects metric choice and renders distribution.",
        instruction=_metric_instruction,
        tools=[metric_distribution, reset_pipeline, exa_mcp_research],
    )

    def should_run_data(ctx: InvocationContext) -> bool:
        return not bool(ctx.session.state.get("data_done"))

    def should_run_signals(ctx: InvocationContext) -> bool:
        return bool(ctx.session.state.get("data_done")) and not bool(ctx.session.state.get("signals_done"))

    def should_run_backtest(ctx: InvocationContext) -> bool:
        return bool(ctx.session.state.get("signals_done")) and not bool(ctx.session.state.get("backtest_done"))

    def should_run_metric(ctx: InvocationContext) -> bool:
        return bool(ctx.session.state.get("backtest_done")) and not bool(ctx.session.state.get("metric_done"))

    pipeline = SequentialAgent(
        name="trading_pipeline",
        description="Runs data fetch, signal generation, backtesting, and metric visualization in order.",
        sub_agents=[
            ConditionalAgent(
                name="data_gate",
                description="Runs data step if needed.",
                wrapped_agent=data_agent,
                should_run=should_run_data,
            ),
            ConditionalAgent(
                name="signals_gate",
                description="Runs signals step if needed.",
                wrapped_agent=signals_agent,
                should_run=should_run_signals,
            ),
            ConditionalAgent(
                name="backtest_gate",
                description="Runs backtest step if needed.",
                wrapped_agent=backtest_agent,
                should_run=should_run_backtest,
            ),
            ConditionalAgent(
                name="metric_gate",
                description="Runs metric step if needed.",
                wrapped_agent=metric_agent,
                should_run=should_run_metric,
            ),
        ],
    )

    return pipeline
