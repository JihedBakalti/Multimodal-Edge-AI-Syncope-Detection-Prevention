'''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Greeting_server")

@mcp.tool()
def say_hello() -> str:
    return "hello, ala"

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''

from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP(name="Math_Server")

# --- Math Tools ---

@mcp.tool()
def add(a: float, b: float) -> float:
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    if b == 0:
        return "Error: Division by zero"
    return a / b

@mcp.tool()
def say_hello() -> str:
    return "Hello, Welcome to Math Server !"

if __name__ == "__main__":
    mcp.run(transport="stdio")
