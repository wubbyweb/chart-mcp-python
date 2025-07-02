"""HTTP server variant for Chart MCP Python."""

import sys
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

from .server.chart_service import chart_service
from .server.storage import storage
from .shared.schema import ChartConfig, InsertChartRequest


# Create both REST API and MCP FastMCP instance
app = FastAPI(
    title="Chart-IMG MCP HTTP Server",
    description="HTTP transport for Chart-IMG MCP server with both REST API and MCP protocol support",
    version="1.0.0",
)

# Import the MCP server instance from main.py
from .main import mcp


class GenerateChartRequest(BaseModel):
    """Request model for chart generation."""
    symbol: str
    interval: str
    chartType: str
    duration: str = "1M"
    width: int = 800
    height: int = 600
    theme: str = "light"
    indicators: List[Dict[str, Any]] = []
    drawings: List[Dict[str, Any]] = []
    showVolume: bool = True
    showGrid: bool = True
    timezone: str = "America/New_York"


class ChartResponse(BaseModel):
    """Response model for chart generation."""
    success: bool
    requestId: str
    message: str
    processingTime: float
    base64Data: str = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    is_configured = chart_service.is_configured()
    
    return {
        "status": "healthy" if is_configured else "configuration_error",
        "chart_api": "configured" if is_configured else "api_key_missing",
        "timestamp": datetime.now().isoformat(),
        "message": "All systems operational" if is_configured else "Please set CHART_IMG_API_KEY environment variable",
    }


@app.post("/initialize_chart", response_model=ChartResponse)
async def initialize_chart(request: GenerateChartRequest):
    """Generate a chart via HTTP API."""
    try:
        # Validate chart configuration
        config = ChartConfig(
            symbol=request.symbol,
            interval=request.interval,
            chartType=request.chartType,
            duration=request.duration,
            width=request.width,
            height=request.height,
            theme=request.theme,
            indicators=[],
            drawings=[],
            showVolume=request.showVolume,
            showGrid=request.showGrid,
            timezone=request.timezone,
        )

        # Create chart request
        insert_request = InsertChartRequest(
            symbol=config.symbol,
            interval=config.interval,
            chartType=config.chartType,
            duration=config.duration,
            width=config.width,
            height=config.height,
            indicators=request.indicators,
            drawings=request.drawings,
            theme=config.theme,
            showVolume=config.showVolume,
            showGrid=config.showGrid,
            timezone=config.timezone,
        )
        
        chart_request = await storage.create_chart_request(insert_request)

        # Initialize the chart
        result = await chart_service.initialize_chart(config, chart_request.requestId)

        if result.success:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "completed",
                "chartUrl": result.url,
                "base64Data": result.base64,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            return ChartResponse(
                success=True,
                requestId=chart_request.requestId,
                message=f"Chart initialized successfully for {config.symbol}",
                processingTime=(result.processingTime or 0) / 1000,
                base64Data=result.base64,
            )
        else:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "failed",
                "errorMessage": result.error,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            raise HTTPException(status_code=500, detail=f"Chart initialization failed: {result.error}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing chart: {str(e)}")


@app.get("/chart_status/{request_id}")
async def get_chart_status(request_id: str):
    """Get chart generation status."""
    try:
        chart_request = await storage.get_chart_request(request_id)
        
        if not chart_request:
            raise HTTPException(status_code=404, detail="Chart request not found")

        response_data = {
            "requestId": chart_request.requestId,
            "symbol": chart_request.symbol,
            "interval": chart_request.interval,
            "chartType": chart_request.chartType,
            "duration": chart_request.duration,
            "status": chart_request.status,
            "createdAt": chart_request.createdAt.isoformat(),
        }

        if chart_request.completedAt:
            response_data["completedAt"] = chart_request.completedAt.isoformat()
        
        if chart_request.processingTime:
            response_data["processingTime"] = chart_request.processingTime / 1000
        
        if chart_request.errorMessage:
            response_data["errorMessage"] = chart_request.errorMessage

        if chart_request.status == "completed" and chart_request.base64Data:
            response_data["base64Data"] = chart_request.base64Data

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chart status: {str(e)}")


@app.get("/symbols")
async def get_available_symbols():
    """Get available trading symbols."""
    try:
        symbols = await chart_service.get_available_symbols()
        
        return {
            "symbols": symbols,
            "total": len(symbols),
            "message": "Use symbols in EXCHANGE:SYMBOL format (e.g., NASDAQ:AAPL, NYSE:TSLA, BINANCE:BTCUSDT)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting symbols: {str(e)}")


@app.get("/recent_requests")
async def get_recent_requests(limit: int = 10):
    """Get recent chart generation requests."""
    try:
        requests = await storage.get_recent_chart_requests(limit)

        return {
            "requests": [
                {
                    "requestId": req.requestId,
                    "symbol": req.symbol,
                    "interval": req.interval,
                    "chartType": req.chartType,
                    "duration": req.duration,
                    "status": req.status,
                    "createdAt": req.createdAt.isoformat(),
                }
                for req in requests
            ],
            "count": len(requests),
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent requests: {str(e)}")


# MCP tool implementation functions
async def call_initialize_chart(arguments: Dict[str, Any]) -> str:
    """Initialize chart MCP tool implementation."""
    try:
        # Validate the chart configuration
        config = ChartConfig(
            symbol=arguments.get("symbol"),
            interval=arguments.get("interval"),
            chartType=arguments.get("chartType"),
            duration=arguments.get("duration", "1M"),
            width=arguments.get("width", 800),
            height=arguments.get("height", 600),
            theme=arguments.get("theme", "light"),
            indicators=[],
            drawings=[],
            showVolume=arguments.get("showVolume", True),
            showGrid=arguments.get("showGrid", True),
            timezone=arguments.get("timezone", "America/New_York"),
        )

        # Create chart request
        insert_request = InsertChartRequest(
            symbol=config.symbol,
            interval=config.interval,
            chartType=config.chartType,
            duration=config.duration,
            width=config.width,
            height=config.height,
            indicators=arguments.get("indicators", []),
            drawings=arguments.get("drawings", []),
            theme=config.theme,
            showVolume=config.showVolume,
            showGrid=config.showGrid,
            timezone=config.timezone,
        )
        
        chart_request = await storage.create_chart_request(insert_request)

        # Initialize the chart
        result = await chart_service.initialize_chart(config, chart_request.requestId)

        # Update the chart request with the result
        if result.success:
            await storage.update_chart_request(chart_request.requestId, {
                "status": "completed",
                "chartUrl": result.url,
                "base64Data": result.base64,
                "processingTime": result.processingTime,
                "completedAt": datetime.now(),
            })

            response_text = (
                f"Chart initialized successfully!\n\n"
                f"Request ID: {chart_request.requestId}\n"
                f"Symbol: {config.symbol}\n"
                f"Interval: {config.interval}\n"
                f"Chart Type: {config.chartType}\n"
                f"Duration: {config.duration}\n"
                f"Processing Time: {(result.processingTime or 0) / 1000:.1f}s\n\n"
                f"The chart has been initialized and is available as a PNG image."
            )

            if result.base64:
                # Return with base64 image data
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


async def call_get_chart_status(arguments: Dict[str, Any]) -> str:
    """Get chart status MCP tool implementation."""
    try:
        request_id = arguments.get("requestId")
        chart_request = await storage.get_chart_request(request_id)
        
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


async def call_get_available_symbols() -> str:
    """Get available symbols MCP tool implementation."""
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


async def call_get_recent_requests(arguments: Dict[str, Any]) -> str:
    """Get recent requests MCP tool implementation."""
    try:
        limit = arguments.get("limit", 10)
        requests = await storage.get_recent_chart_requests(limit)

        if not requests:
            return "No recent requests found."

        lines = [
            f"Recent Chart Requests ({len(requests)} of {limit})",
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


async def call_health_check() -> str:
    """Health check MCP tool implementation."""
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


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP protocol endpoint for @modelcontextprotocol/inspector compatibility."""
    try:
        # Get the JSON body
        body = await request.json()
        
        # Process MCP request using FastMCP
        # This creates a fake stdio-like environment for HTTP transport
        import json
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture the MCP response
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Create a mock stdin with the request
        request_line = json.dumps(body) + "\n"
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Import the MCP tool implementations directly
            pass
            
            # Handle different MCP method calls
            if body.get("method") == "tools/list":
                # Import the MCP instance to get all registered tools
                from .main import mcp
                
                # Get all tools from the MCP instance
                mcp_tools = []
                
                # Add all the registered tools from the main MCP server
                tools_dict = await mcp.get_tools()
                for tool_name, tool_obj in tools_dict.items():
                    # Convert FastMCP tool to MCP protocol format
                    mcp_tool = tool_obj.to_mcp_tool()
                    
                    # Flatten the schema for MCP Inspector compatibility
                    input_schema = mcp_tool.inputSchema
                    
                    # If the schema has an 'args' property that references a definition,
                    # flatten it to use the definition directly
                    properties = input_schema.get("properties", {})
                    args_prop = properties.get("args", {})
                    
                    if args_prop.get("$ref") and input_schema.get("$defs"):
                        # Extract the referenced definition name
                        ref = args_prop["$ref"]
                        def_name = ref.split("/")[-1]  # Get 'RSIArgs' from '#/$defs/RSIArgs'
                        
                        if def_name in input_schema["$defs"]:
                            # Use the definition directly as the input schema
                            input_schema = input_schema["$defs"][def_name]
                    
                    mcp_tools.append({
                        "name": mcp_tool.name,
                        "description": mcp_tool.description,
                        "inputSchema": input_schema
                    })
                
                tools = mcp_tools
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": tools
                    }
                })
            
            elif body.get("method") == "tools/call":
                # Call a specific tool
                params = body.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                # Import the MCP instance to call tools
                from .main import mcp
                
                # Check if this is a registered MCP tool
                tools_dict = await mcp.get_tools()
                if tool_name in tools_dict:
                    try:
                        # Get the tool and run it with the arguments
                        tool_obj = tools_dict[tool_name]
                        # FastMCP tools expect arguments wrapped in an 'args' parameter
                        wrapped_args = {"args": arguments}
                        tool_result = await tool_obj.run(wrapped_args)
                        
                        # Extract text from the result (FastMCP returns TextContent objects)
                        if isinstance(tool_result, list) and len(tool_result) > 0:
                            result = tool_result[0].text
                        else:
                            result = str(tool_result)
                                
                    except Exception as e:
                        return JSONResponse({
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Tool execution error: {str(e)}"
                            }
                        }, status_code=500)
                else:
                    return JSONResponse({
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {tool_name}"
                        }
                    }, status_code=404)
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                })
            
            elif body.get("method") == "initialize":
                # Initialize the MCP session
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "chart-mcp-python",
                            "version": "1.0.0"
                        }
                    }
                })
            
            else:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {body.get('method')}"
                    }
                }, status_code=404)
    
    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() else None,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)


def main() -> None:
    """Run the HTTP server."""
    try:
        print("Starting Chart-IMG MCP HTTP Server on http://localhost:3001", file=sys.stderr)
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=3001,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nShutting down Chart-IMG MCP HTTP Server", file=sys.stderr)
    except Exception as e:
        print(f"HTTP Server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()