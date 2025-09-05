#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for SSB Agent testing
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    env_vars = {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-api-key-12345",
        "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
        "AZURE_OPENAI_MODEL": "gpt-5",
        "AZURE_REASONING_EFFORT": "medium",
        "AZURE_REASONING_SUMMARY": "auto",
        "MCP_REQUEST_TIMEOUT_S": "30",
        "MCP_CONNECT_TIMEOUT_S": "15"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_ssb_table():
    """Sample SSB table data for testing"""
    return {
        "id": "07459",
        "label": "Population by municipality",
        "description": "Population statistics by municipality, gender and age",
        "updated": "2024-12-01T10:30:00",
        "firstPeriod": "2020",
        "lastPeriod": "2024",
        "variableNames": ["Region", "Kjonn", "Alder", "ContentsCode", "Tid"],
        "subjectCode": "02",
        "path": ["Population", "Population by municipality"]
    }

@pytest.fixture
def sample_table_metadata():
    """Sample table metadata response"""
    return {
        "label": "Population by municipality",
        "description": "Population statistics by municipality, gender and age",
        "updated": "2024-12-01T10:30:00",
        "source": "Statistics Norway (SSB)",
        "dimension": {
            "Region": {
                "category": {
                    "index": ["0301", "1103", "3806", "4601", "5001"],
                    "label": {
                        "0301": "Oslo",
                        "1103": "Stavanger", 
                        "3806": "Bergen",
                        "4601": "Tromsø",
                        "5001": "Trondheim"
                    }
                },
                "extension": {
                    "codeLists": [
                        {
                            "id": "vs_Fylker",
                            "label": "Counties (fylker)",
                            "type": "valueset"
                        },
                        {
                            "id": "vs_Kommuner",
                            "label": "Municipalities (kommuner)",
                            "type": "valueset"
                        },
                        {
                            "id": "agg_Fylker2024",
                            "label": "County structure 2024",
                            "type": "aggregation"
                        }
                    ]
                }
            },
            "Kjonn": {
                "category": {
                    "index": ["0", "1", "2"],
                    "label": {
                        "0": "Both sexes",
                        "1": "Males",
                        "2": "Females"
                    }
                },
                "extension": {"codeLists": []}
            },
            "Alder": {
                "category": {
                    "index": ["000", "001-004", "005-009", "010-014", "080-089", "090-099", "100+"],
                    "label": {
                        "000": "0 years",
                        "001-004": "1-4 years",
                        "005-009": "5-9 years", 
                        "010-014": "10-14 years",
                        "080-089": "80-89 years",
                        "090-099": "90-99 years",
                        "100+": "100 years or more"
                    }
                },
                "extension": {"codeLists": []}
            },
            "ContentsCode": {
                "category": {
                    "index": ["Folkemengde", "Innflytting", "Utflytting"],
                    "label": {
                        "Folkemengde": "Population",
                        "Innflytting": "Immigration",
                        "Utflytting": "Emigration"
                    }
                },
                "extension": {"codeLists": []}
            },
            "Tid": {
                "category": {
                    "index": ["2020", "2021", "2022", "2023", "2024"],
                    "label": {
                        "2020": "2020",
                        "2021": "2021",
                        "2022": "2022",
                        "2023": "2023",
                        "2024": "2024"
                    }
                },
                "extension": {"codeLists": []}
            }
        }
    }

@pytest.fixture
def sample_filtered_data():
    """Sample filtered data response"""
    return {
        "label": "Population by municipality",
        "source": "Statistics Norway (SSB)",
        "updated": "2024-12-01T10:30:00",
        "dimension": {
            "Region": {
                "category": {
                    "label": {
                        "0301": "Oslo",
                        "1103": "Stavanger",
                        "3806": "Bergen"
                    }
                }
            },
            "Kjonn": {
                "category": {
                    "label": {"0": "Both sexes"}
                }
            },
            "Alder": {
                "category": {
                    "label": {"000": "Total"}
                }
            },
            "ContentsCode": {
                "category": {
                    "label": {"Folkemengde": "Population"}
                }
            },
            "Tid": {
                "category": {
                    "label": {"2024": "2024"}
                }
            }
        },
        "value": [695000, 148000, 285000]  # Oslo, Stavanger, Bergen populations
    }

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API calls"""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock()
        
        mock_client.return_value.__aenter__.return_value.get = MagicMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value.post = MagicMock(return_value=mock_response)
        
        yield mock_client, mock_response

@pytest.fixture
def norway_regions():
    """Common Norwegian regions for testing"""
    return {
        "municipalities": {
            "0301": "Oslo",
            "1103": "Stavanger", 
            "3806": "Bergen",
            "4601": "Tromsø",
            "5001": "Trondheim"
        },
        "counties": {
            "03": "Oslo",
            "11": "Rogaland",
            "38": "Vestland", 
            "46": "Troms og Finnmark",
            "50": "Trøndelag"
        }
    }

@pytest.fixture
def common_queries():
    """Common query patterns for testing"""
    return {
        "population": [
            "befolkning",
            "population", 
            "innbyggere",
            "folkemengde"
        ],
        "employment": [
            "sysselsetting",
            "employment",
            "arbeid",
            "jobber"
        ],
        "education": [
            "utdanning",
            "education",
            "skole",
            "universitet"
        ],
        "complex": [
            "befolkning oslo stavanger sammenligning",
            "unemployment rates by region over time",
            "education levels trondheim bergen comparison"
        ]
    }

@pytest.mark.asyncio
async def async_test_wrapper(test_func):
    """Wrapper for async test functions"""
    if asyncio.iscoroutinefunction(test_func):
        await test_func()
    else:
        test_func()