# Simple SSB - Norwegian Statistics API Agent

An intelligent Norwegian statistics agent that efficiently queries SSB (Statistics Norway) data using Azure OpenAI and MCP (Model Context Protocol) capabilities.

## 🎯 Overview

Simple SSB provides an intelligent interface to Norwegian statistical data through:
- **Advanced SSB API Integration**: Direct access to Statistics Norway's PxWeb API v2-beta
- **Smart Aggregation**: Automatic use of county/municipality groupings and administrative levels
- **Efficient Querying**: Optimized workflow with minimal API calls (typically 3-4 calls per query)
- **Natural Language Interface**: Ask questions in Norwegian or English
- **MCP Architecture**: Built using Model Context Protocol for extensible AI tool integration

## ✨ Key Features

### 🚀 Intelligent Query Processing
- **Adaptive Table Discovery**: Finds relevant SSB tables using advanced search syntax
- **Automatic Dimension Mapping**: Translates Norwegian display names to API dimension names
- **Smart Aggregation**: Uses SSB's built-in county/municipality groupings automatically
- **Comparison Queries**: Efficiently handles "which X has most Y" questions in single API calls

### 📊 Advanced SSB API Features
- **Code Lists & Aggregation**: Automatic discovery and use of `agg_Fylker2024`, `vs_Kommune`, etc.
- **Time Selections**: Advanced time filtering with `top(5)`, `range(2020,2023)`, wildcards
- **Geographic Filtering**: County and municipality level data with proper aggregation
- **Error Recovery**: Intelligent error handling with automatic dimension discovery

### 🎨 Rich Output Display
- **Formatted Tables**: Clean, readable data presentation
- **Comparison Highlighting**: Automatic identification of maximum/minimum values
- **Summary Statistics**: Quick insights with winner identification
- **Progress Tracking**: Real-time tool usage and reasoning display

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI API access
- Required Python packages (see requirements.txt)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/trygvels/simple-ssb.git
cd simple-ssb
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Azure OpenAI**:
Create a `.env` file with your Azure OpenAI credentials:
```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_O3MINI_DEPLOYMENT=your-o3-mini-deployment-name
```

## 🚀 Usage

### Command Line Interface

**Basic population query**:
```bash
python ssb_agent_mcp.py "Hvilket fylke har flest folk?"
```

**English queries**:
```bash
python ssb_agent_mcp.py "What is the population of Norway?"
python ssb_agent_mcp.py "Show me unemployment statistics by region"
```

**Comparison queries**:
```bash
python ssb_agent_mcp.py "Sammenlign befolkning mellom fylker"
python ssb_agent_mcp.py "Which county has the highest income?"
```

**Time-based queries**:
```bash
python ssb_agent_mcp.py "Befolkningsutvikling siden 2020"
python ssb_agent_mcp.py "Latest employment statistics"
```

### Example Output

```
🧠 o3-mini Reasoning Model Analysis

🔧 Calling: search_tables_advanced
📤 search_tables_advanced Result:
🔍 Found 10 tables for query: 'befolkning fylke'

🔧 Calling: analyze_table_structure  
📊 Table Analysis: 03031
📋 Available Dimensions (6):
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Dimension    ┃ Label        ┃ Values    ┃ Aggregation  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Region       │ Region       │ 41 values │ ✅ Available │
└──────────────┴──────────────┴───────────┴──────────────┘

🔧 Calling: get_filtered_data
📈 Data Retrieved from 03031
📊 Summary:
  • Winner: Akershus with 740,680 people
```

## 🎯 Query Examples

### Population Queries
```bash
# Norwegian
python ssb_agent_mcp.py "Hvor mange bor i Oslo?"
python ssb_agent_mcp.py "Befolkning per fylke 2025"

# English  
python ssb_agent_mcp.py "Population of Bergen"
python ssb_agent_mcp.py "Which municipality has most people?"
```

### Economic Queries
```bash
python ssb_agent_mcp.py "Arbeidsledighet per fylke"
python ssb_agent_mcp.py "Income statistics by age group"
python ssb_agent_mcp.py "Employment trends since 2020"
```

### Comparison Queries
```bash
python ssb_agent_mcp.py "Hvilket fylke har høyest inntekt?"
python ssb_agent_mcp.py "Compare education levels between counties"
python ssb_agent_mcp.py "Which industry employs most people?"
```

## 📚 Documentation

- **[Development Instructions](INSTRUCTIONS.md)**: Comprehensive development guide and technical documentation
- **[SSB API Guide](SSB_API_GUIDE.md)**: Comprehensive SSB API documentation
- **[MCP Agents Guide](MCP_AGENTS_GUIDE.md)**: MCP integration and tool development
- **[Requirements](requirements.txt)**: Python dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Statistics Norway (SSB)** for providing comprehensive statistical data APIs
- **Azure OpenAI** for advanced language model capabilities
- **MCP (Model Context Protocol)** for extensible AI tool integration
- **FastMCP** for efficient MCP server implementation

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check existing documentation in the project files
- Review the SSB API documentation for data-specific questions

---

**Built with ❤️ for Norwegian statistics and data analysis** 