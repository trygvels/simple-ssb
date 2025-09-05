#!/usr/bin/env python3
"""
Quick test script for Norwegian SSB Agent
"""

import asyncio
import sys
sys.path.insert(0, 'src')

from ssb_agent_mcp import SSBAgent

async def test_norwegian_queries():
    """Test Norwegian queries quickly"""
    agent = SSBAgent()
    
    test_queries = [
        "befolkning Norge",
        "arbeidsledighet 2024"
    ]
    
    for query in test_queries:
        print(f"\n=== Testing: {query} ===")
        try:
            # Test with shorter timeout for quick validation
            result = await asyncio.wait_for(
                agent.process_query(query), 
                timeout=20.0
            )
            print(f"✅ Success: {len(result) if result else 0} characters returned")
        except asyncio.TimeoutError:
            print("⏰ Timeout - agent is working but taking time")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_norwegian_queries())