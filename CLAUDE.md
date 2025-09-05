# SSB Agent MCP - AI Assistant Instructions

## System Overview

An intelligent MCP-based interface for Norwegian statistics (SSB) that uses self-learning patterns to access data without hardcoded assumptions. The system combines Azure OpenAI's gpt-5/gpt-5-mini with 9 domain-agnostic MCP tools.

## Core Architecture

**Components:**
- `src/mcp_server.py`: FastMCP server with 9 intelligent tools for SSB API access
- `src/ssb_agent_mcp.py`: Azure OpenAI agent with pattern learning capabilities
- Domain-agnostic design: No hardcoded table IDs, dimensions, or domain-specific logic

**Key Tools:**
1. `search_tables_advanced` - Find relevant tables
2. `analyze_table_structure` - Discover dimensions
3. `discover_dimension_values` - Get available codes
4. `get_filtered_data` - Retrieve data
5. `diagnose_table_requirements` - Learn from errors

## Critical Development Rules

### 🚫 NEVER Add Domain-Specific Examples

**FORBIDDEN:**
- Hardcoded table IDs or dimension names
- Domain-specific workflows (employment, population, etc.)
- Fixed example queries with specific parameters

**REQUIRED:**
- Generic, domain-agnostic patterns only
- Dynamic discovery through API responses
- Learn dimension names from actual metadata

**Why:** The system's strength is handling ANY statistical domain without modification.

### ✅ Core Principles

1. **Never Assume** - Always discover from SSB API
2. **Learn from Errors** - Use error messages as learning opportunities
3. **Stay Generic** - No domain-specific hardcoding
4. **Self-Correct** - Apply learned patterns within sessions

### Pattern Learning Examples

```python
# When this fails:
discover_dimension_values("region")  
# Error: Use one of: ['Region', 'ContentsCode', 'Tid']

# Agent learns: "region" → "Region"
# Applies pattern to similar cases in session
```

## Testing Philosophy

- **No Mocked Data**: Test against real SSB API
- **No Hardcoded Paths**: All discovery is dynamic
- **Agent-Compatible Output**: Validate tool outputs work with agent

## File Structure

```
simple-ssb/
├── src/
│   ├── ssb_agent_mcp.py    # Agent orchestration
│   └── mcp_server.py        # MCP tools
├── tests/
│   ├── test_mcp_tools.py   # Tool validation
│   └── test_agent_integration.py  # Integration tests
└── CLAUDE.md                # This file
```

## Key Behaviors

1. **Error Recovery**: Transform API errors into correct calls
2. **Pattern Memory**: Remember dimension mappings within sessions
3. **Adaptive Workflow**: Adjust strategy based on table structure
4. **Self-Documentation**: Tools provide clear error guidance

## Success Indicators

✅ Works with any SSB statistical domain
✅ No hardcoded assumptions
✅ Learns from every interaction
✅ Provides clear, sourced answers
✅ Handles errors gracefully

## When Modifying Code

**DO:**
- Maintain domain agnosticism
- Preserve dynamic discovery
- Keep error messages informative
- Test with real API responses

**DON'T:**
- Add specific table IDs
- Hardcode dimension names
- Create domain-specific logic
- Use simulated test data

## Model Configuration

- **Model**: gpt-5/gpt-5-mini
- **API**: Azure OpenAI with Responses API
- **Reasoning**: Configurable effort levels
- **Streaming**: Full event handling support

---

*Remember: This system's power comes from being completely generic. Every query type - population, employment, education, healthcare - uses the same intelligent discovery process.*