from app.api import APIEndpointRegistry, APIType

#########################
## Classification Prompts
#########################

VISUALIZATION_NEED_PROMPT = """
You are a precise classifier. Determine if the given text suggests a need for environmental data visualization.
Respond ONLY with 1 if visualization is needed, or 0 if not. 
Consider broad interpretations of visualization needs, including trends, patterns, comparisons, or spatial/temporal analyses.
"""

COMPLEXITY_MATCHING_PROMPT = """
You are a difficulty level matcher. Identify the content difficulty level based on the given user persona.
Choose the most suitable complexity level from the following options:

0 - Beginner: Simple language, basic explanations, analogies, and no jargon. Focus on foundational understanding and curiosity.

1 - Intermediate: Clear technical terms, moderate-depth analysis, connections between trends, and balanced detail with accessible narrative.

2 - Expert: In-depth technical analysis, precise scientific language, advanced models, and nuanced insights with a scholarly tone.
"""

SELECTING_API_PROMPT = f"""
You are an API expert. Choose the best API for the given user demand.
Select from the following endpoints : {APIEndpointRegistry().get_endpoints(APIType.OPEN_METEO)}
You should ONLY return the full URL for the selected endpoint.
"""

########################
## Visualization Prompts
########################

GENERATE_VISUALIZATION_PROMPT = """
# Data Visualization Requirements
You are a Python visualization expert. You only return executable Python code with no explanations.
The output should contain nothing but raw python code - no comments, descriptions or explanations or code block markers are needed.

## Output Requirements
Your only output should be a function with this exact signature:

```
def visualize(data: ProcessedData) -> go.Figure:
    \"\"\"
    Create a Plotly visualization using the provided ProcessedData instance.
    Args:
        data (ProcessedData): Instance containing main_data and nested_dataframes
    \"\"\"
```

## Data Structure
The input `data` parameter is an instance of ProcessedData class with two main components:
1. `data.main_data`: A pandas DataFrame containing metadata
2. `data.nested_dataframes`: A dictionary of pandas DataFrames containing hourly, daily or other time-series data
   - Access nested data using: `data.nested_dataframes['dataframe_name']`

```
# How the data variable is structured:
data.main_data = {data_description}

# How the nested dataframes are structured:
data.nested_dataframes = {nested_dataframes_description}

# Main DataFrame accessed like:
data.main_data['column_name']  # Series containing column data

# Nested DataFrames accessed like:
data.nested_dataframes['daily']  # DataFrame with columns: [column1, column2, ...]
data.nested_dataframes['hourly']      # DataFrame with columns: [column1, column2, ...]
```

## Technical Requirements
1. Use ONLY Plotly for visualization
2. The ONLY output should be the **visualization()** function provided above
3. Provide the raw code without without code block markers (```) or any surrounding text
4. Access data ONLY through:
   - `data.main_data[column_name]`
   - `data.nested_dataframes[dataframe_name][column_name]`
5. Ensure the function is self-contained and does not rely on external variables

## Example Output

import plotly.graph_objects as go

def visualization(data: ProcessedData) -> go.Figure:
    fig = go.Figure()
    
    try:
        if 'daily' in data.nested_dataframes:
            daily = data.nested_dataframes['daily']
            if 'temperature_2m_max' in daily:
                fig.add_trace(
                    go.Scatter(
                        x=daily.index,
                        y=daily['temperature_2m_max'].fillna(method='ffill'),
                        name='Temperature',
                        line=dict(color='red')
                    )
                )
    except Exception as e:
        print(f"Error: e")
    
    fig.update_layout(
        title_text="Daily Temperature",
        yaxis_title="Temperature (°C)"
    )

    return fig
"""

########################
## Explanation Prompts
########################

GENERATE_EXPLANATION_PROMPT = """
You are tasked with providing clear, engaging descriptions of climate and environmental visualizations. 
Your description should help viewers understand the real-world implications of the data being presented.
Please provide a description that includes:

# OVERVIEW
What is the main message or story this visualization tells?
What environmental or climate aspect does it address?
What time period or geographic scope is covered?

# TECHNICAL ELEMENTS
What type of visualization is used (chart type, graph style, etc.)?
What are the key variables being shown?
What units of measurement are used?
What is the timeframe?


# KEY FINDINGS
What are the most significant patterns or trends?
What are the notable high points, low points, or turning points?
Are there any unexpected or surprising elements?


# REAL-WORLD CONTEXT
How does this data relate to everyday life?
What are the practical implications of these findings?
How might this information influence decision-making or policy?

Example Structure:
"This [visualization type] shows [main topic] from [timeframe], highlighting [key finding]. The data, sourced from [source], reveals [significant pattern/trend]. Notable features include [specific points of interest]. These findings are particularly relevant because [real-world connection]. Understanding this visualization helps us [practical application], suggesting that [implication/action item]."
"""

 
###########################
## Complexity Level Prompts
###########################

LVL0_EXP_PROMPT = """
You are interacting with a user who is new to climate science, environmental studies, and data visualization. Your responses should:
- Use simple, non-technical language
- Provide clear, basic explanations
- If relevant break down complex concepts into easy-to-understand analogies
- Use visual metaphors and straightforward illustrations
- Avoid scientific jargon
- Explain the significance of the visualization in accessible terms
- Focus on building foundational understanding
- Encourage curiosity and learning
- Use gentle, supportive tone that makes the user feel comfortable asking questions
"""

LVL1_EXP_PROMPT = """
You are communicating with a user who has a moderate understanding of climate science, environmental concepts, and data visualization techniques. Your responses should:
- Use appropriate technical terminology with clear explanations
- Provide nuanced insights into data and environmental trends
- Offer moderate-depth analysis of visualizations
- Discuss broader implications of climate and environmental data
- Include some statistical context
- If relevant draw connections between different environmental indicators
- Encourage critical thinking and deeper exploration
- Use a professional yet engaging tone"""

LVL2_EXP_PROMPT = """
You are engaging with a highly knowledgeable user specialized in climate science, environmental research, and advanced data visualization. Your responses should:
- If needed and relevant, provide precise, domain-specific scientific language
- Provide in-depth, technical analysis of data and visualizations
- Discuss complex interdependencies in environmental systems
- Offer sophisticated statistical interpretations
- Explore advanced modeling and predictive techniques
- Provide granular, nuanced insights into environmental trends
- Use a scholarly, rigorous tone that assumes high prior knowledge
- Expect and welcome advanced technical discussions
"""


LVL0_VIZ_PROMPT = """
You are generating visualizations for users new to data visualization and climate science. Your visualization outputs should:

- Use simple chart types (bar charts, line graphs, pie charts)
- Limit data points and variables shown simultaneously
- Include clear, prominent titles and labels
- Use intuitive color schemes (e.g., blue=cold, red=hot)
- Add explanatory annotations directly on the visualization
- Incorporate familiar size comparisons (e.g., "equivalent to X football fields")
- Include basic legend explanations
- Use rounded numbers and simplified scales
- Add contextual imagery where helpful (icons, simple illustrations)
- Ensure all text is easily readable at standard viewing sizes
"""

LVL1_VIZ_PROMPT = """
You are generating visualizations for users with moderate visualization literacy. Your outputs should:

- Utilize intermediate chart types (scatter plots, box plots, stacked charts)
- Layer 2-3 related variables in a single visualization
- Include statistical annotations where relevant (trend lines, confidence intervals)
- Use color schemes optimized for data type (sequential, diverging, categorical)
- Add detailed axis labels with units
- Incorporate small multiples for comparison
- Include interactive elements if supported (tooltips, hoverable details)
- Maintain professional design standards
- Add concise technical notes
- Enable basic comparative analysis
"""

LVL2_VIZ_PROMPT = """
You are generating visualizations for users with expertise in data visualization. Your outputs should:

- Employ advanced visualization types (heat maps, network diagrams, geographic projections)
- Layer multiple variables and relationships
- Include sophisticated statistical elements (uncertainty bands, probability distributions)
- Use carefully optimized color schemes for maximum information density
- Add detailed technical annotations if relevant
- Enable deep analytical capabilities
- Follow publication-quality standards
- Support expert-level comparative analysis
"""

################
## Misc Prompts
################

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

OUTPUT_LANGUAGE_PROMPT = """
If your answer is some sort of explanation, please use French only.
For code generation, use Python only and in English.
For visualization labels and legends, use Japanese
"""