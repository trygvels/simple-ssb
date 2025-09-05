# Improved SSB Agent Instructions

Based on behavior analysis, these instructions focus on eliminating common inefficiencies.

## 🔧 TOOL USAGE OPTIMIZATION

### **CRITICAL RULES FOR get_filtered_data:**
1. **filters parameter is ALWAYS REQUIRED** - never call without it
2. **Time goes in filters, not time_selection**: `{"Tid": "2024"}` not `time_selection="2024"`
3. **Use exact dimension names from get_table_info** - never guess
4. **For comparisons, use wildcards**: `{"Region": "*"}` gets all regions in one call

### **SEARCH STRATEGY (Limit to 2 calls maximum):**
- Call 1: Primary Norwegian terms (e.g., "sysselsatte næring")  
- Call 2: Alternative terms if needed (e.g., "arbeidstakere NACE")
- STOP after 2 searches - don't keep trying variations

### **FILTERING PATTERNS BY QUERY TYPE:**

#### **Ranking/Top N Queries** (e.g., "top 5 næringer"):
```
1. search_tables (1-2 calls max)
2. get_table_info (1 call)
3. get_filtered_data with wildcard on comparison dimension:
   filters={
     "Region": "0",           # Whole country
     "NACE2007": "*",         # All industries 
     "Tid": "2024"            # Specific year
   }
   code_lists={"NACE2007": "agg_NACE2007arb11"}  # Use aggregation
```

#### **Simple Count Queries** (e.g., "befolkning Norge"):
```
1. search_tables (1 call)
2. get_table_info (1 call)  
3. get_filtered_data:
   filters={
     "Region": "0",
     "ContentsCode": "Folkemengde",
     "Tid": "2024"
   }
```

#### **Regional Comparison** (e.g., "Oslo vs Bergen"):
```
1. search_tables (1 call)
2. get_table_info (1 call)
3. get_filtered_data with wildcard on Region:
   filters={
     "Region": "*",           # All regions
     "ContentsCode": "xxx",
     "Tid": "2024"
   }
   # Agent can then filter Oslo/Bergen from results
```

### **ERROR PREVENTION:**
- **Before calling get_filtered_data**: Always have a complete filters dict ready
- **Use get_table_info results**: Copy exact dimension names, don't modify them
- **Wildcard first**: Try `"*"` before discovering specific dimension values
- **One data call goal**: Aim to get all needed data in a single get_filtered_data call

### **WHEN TO USE discover_dimension_values:**
- Only when wildcards won't work (e.g., need specific municipality codes)
- Skip it for ranking queries - wildcards are better
- Skip it when code_lists provide the grouping you need

## 🎯 EXPECTED PERFORMANCE:
- **Tool calls per query**: 3-5 (not 7-8)
- **get_filtered_data calls**: 1-2 maximum  
- **Search calls**: 1-2 maximum
- **Zero validation errors**: Always provide complete parameters