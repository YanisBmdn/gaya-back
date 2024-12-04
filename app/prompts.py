from app.api import APIEndpointRegistry, APIType

CLASSIFYING_PROMPT = """
You are a precise classifier. Determine if the given text suggests a need for environmental data visualization.
Respond ONLY with 1 if visualization is needed, or 0 if not. 
Consider broad interpretations of visualization needs, including trends, patterns, comparisons, or spatial/temporal analyses.
"""


SELECTING_API_PROMPT = f"""
You are an API expert. Choose the best API for the given user demand.
Select from the following endpoints : {APIEndpointRegistry().get_endpoints(APIType.OPEN_METEO)}
You should ONLY return the full URL for the selected endpoint.
"""

GENERATE_VISUALIZATION_PROMPT = """
# Data Visualization Requirements:
1. Use ONLY fields specified in the data dictionary
2. DYNAMICALLY process input data
3. Create Plotly visualization
4. NO hardcoded values or assumptions
5. MUST be directly executable

# Data Dictionary Sample and Description:
{data_description}

# !!STRICT CONSTRAINTS!!:
- Provide a visualization function called ```visualize(data)``` that takes the data dictionary as input
- Use provided data fields EXACTLY
- Adaptive visualization based on data structure
- Handle potential null/missing values
"""

BUILD_EXTERNAL_QUERY_PROMPT = """
Provide the full URL with accurate parameters based on the user prompt for this API endpoint : ```{api_endpoint}```.
If the user hasn't specified it, base the location in Nagoya, Japan.
If not specified the timescale should be the past 2 years.
Here are some examples with the url and parameters:

Air Quality in Nagoya
```https://air-quality-api.open-meteo.com/v1/air-quality?latitude=35.1815&longitude=136.9064&hourly=pm10,pm2_5```

Temperatures in Fuji for the past 10 years
```https://archive-api.open-meteo.com/v1/archive?latitude=35.1667&longitude=138.6833&start_date=2014-11-17&end_date=2024-12-01&hourly=temperature_2m```



Here is the parameters documentation for the endpoint: 
{api_endpoint_parameters}
"""