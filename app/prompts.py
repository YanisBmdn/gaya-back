#########################
## Classification Prompts
#########################

VISUALIZATION_NEED_PROMPT = """
You are a precise classifier. Determine if the given prompt could use an environmental data visualization to help explain things.
Set the need_visualization variable ONLY to 1 if visualization is needed, or 0 if not.
Set the topic_of_interest variable to the main topic of interest mentioned in the text. It MUST be related to climate change (e.g., temperature trends, air quality, precipitation).
Consider broad interpretations of visualization needs, including trends, patterns, comparisons, or spatial/temporal analyses.
"""

COMPLEXITY_MATCHING_PROMPT = """
You are a difficulty level matcher. Identify the content difficulty level based on the given user description.
Choose the most suitable complexity level from the following options:
0 - Foundational: Simple everyday language, visual examples, relatable stories, and connections to daily life. Designed for children, seniors, and those with no prior knowledge.
1 - Informational: Balanced mix of accessible explanations and some technical concepts, practical applications, and moderate detail. Appropriate for general public with basic awareness.
2 - Comprehensive: Scientific content with some terminology, data-based discussions, and policy implications. Suitable for knowledgeable citizens with strong interest in climate issues.
"""

########################
## Visualization Prompts
########################

SCENARIO_EXPLANATION = """
The user is participating in a civic decision-making experiment about budget allocation. They are presented with a local government budget allocation scenario that requires them to make an evidence-based choice.

SCENARIO DETAILS:
{scenario}

DECISION OPTIONS:
{options}

Your role is to provide information to the user to help him understand and comprehend climate data to promote fact-based decision making. Do not advocate for any specific option - remain neutral while providing factual information.
"""

DETERMINE_VISUALIZATION_TYPE_PROMPT = """
Your task is to recommend a climate change visualization using OpenMeteo API data.
The visualization will ONLY use Open-Meteo data, which includes things such as current weather data, historical weather data, and weather forecasts including temperature, precipitation, wind speed/direction, humidity, and air quality parameters (like PM2.5, PM10, and various gases).

Topic: {topic_of_interest}
User Description: {persona}
Complexity Level: {complexity_level}
City / Geographical area of interest {location}

Guidelines:
- Focus on medium to long term patterns or trends that are relevant to climate change analysis
- If relevant, consider having subplots or multiple traces (e.g., comparing different locations)
- Keep it simple and clear for the user's understanding. Avoid overly complex visualizations.
- Match complexity to user expertise level
- Visualization is achievable ONLY with data from OpenMeteo such as temperature and precipitation.
- The visualization shouldn't use any external data sources, assets or icons. Use ONLY OpenMeteo data.

Reference Examples by Complexity:

1. Beginner Level:
Visualization Name: "Yearly Temperature Change"
Chart Type: Line chart
Climate Change Focus: Simple yearly temperature trend
Visual Elements:
- Single line of yearly average temperatures
- X-axis: years (2000-2023)
- Y-axis: temperature in °C

2. Intermediate Level:
Visualization Name: "Pollution comparison between Japan's main cities"
Chart Type: Line chart
Climate Change Focus: Pollution levels in different cities
Visual Elements:
- Multiple lines for different cities
- X-axis: daily time series
- Y-axis: pollution levels (PM2.5, PM10)
- Color-coded lines for each city

3. Expert Level:
Visualization Name: "Heatmap highlighting monthly Temperature Anomalies"
Chart Type: Heatmap
Climate Change Focus: Monthly temperature deviations from baseline
Visual Elements:
- X-axis: months (Jan-Dec)
- Y-axis: years (1980-2023)
- Color scale: temperature anomalies in °C
- Baseline period: 1951-1980
"""

DETERMINE_NEEDED_DATA_PROMPT = """
Your current task is to determine the needed data from OpenMeteo API for a climate visualization. You should only consider OpenMeteo API data.
The visualization type has been defined by another expert as following:
{visualization_type}

And the following available data from OpenMeteo API:
# API Endpoint Information
{API_ENDPOINT_INFORMATION}

The area of interest mentionned by the user is: {location}. It should be used as the default location if not mentionned in the prompt.

Provide three outputs in this format:

# Output Example
needed_data:
Daily mean temperature (temperature_2m_mean)
Time range: 1980-2023
Geographic scope: Single location (London, UK)
Resolution: Daily values to be aggregated to annual


data_processing_steps:
Step 1: Calculate annual average temperatures
Step 2: Compute 5-year moving average
"""

RETRIEVE_DATA_PROMPT = """
Your current task is to retrieve the needed data for a climate visualization.
The visualization type and needed data have already been defined, as following : 

# Visualization Type
{visualization_type}

# Needed Data
{needed_data}

# API Endpoint Information
{API_ENDPOINT_INFORMATION}

# Area of interest
{location}

Your task is to define the API endpoint and inline-parameters to retrieve the required data.
If not mentionned, the location should be set to Nagoya, Japan.
Be careful about the potential amount of data that could be returned (ex. hourly data of 10 years or more isn't acceptable).
DON'T HALLUCINATE ON THE PARAMETERS AND THE DATA. IF DATA ISN'T AVAILABLE IN WHAT WAS PROVIDED, DON'T INCLUDE IT.

# Output Example
Daily minimum, maximum and mean temperature in Nagoya, Japan from 2015-01-18 to 2025-02-01
url="https://archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2015-01-18&end_date=2025-02-01&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean"

Hourly PM10 and PM2.5 concentration in Nagoya, Japan from 2015-01-01 to 2025-01-01
url="https://air-quality-api.open-meteo.com/v1/air-quality?latitude=35.1815&longitude=136.9064&hourly=pm10,pm2_5&start_date=2015-01-01&end_date=2025-01-01"
"""

PROCESS_DATA_PROMPT = """
Your current task is to create a function to process raw climate data for visualization.
You should only return python code that will be then executed with python ```exec()```. Your response shouldn't contain any additional text or comments.
Given:
- Visualization Goal: {visualization_type}
- Processing Plan: {processing_steps}
- Input Data Structure: {data_description}

Here's the output format of the function you need to create:

@dataclass
class ProcessedData:
    A generic container for processed dataframes.
    
    Attributes:
        main_data (pd.DataFrame): The base dataframe with non-nested columns
        nested_dataframes (dict[str, pd.DataFrame]): Dictionary of expanded nested dataframes
    main_data: pd.DataFrame
    nested_dataframes: dict[str, pd.DataFrame]


Create a data processing function with:
def process_raw_data(data: pd.DataFrame) -> ProcessedData:
    '''
    Process raw climate data for visualization.
    '''

Requirements:
1. Handle missing or invalid data
2. Validate input data structure
3. Optimize for performance with large datasets
4. Return clean, visualization-ready DataFrame
5. Add all imports necessary to your code.
5. You should only use pandas and numpy for data processing. No additional libraries are allowed
"""

BUILD_VISUALIZATION_PROMPT = """
Create a visualization function with these parameters:
- Visualization goal: {visualization_type}
- Complexity Level: {complexity_level}
- Processing Steps: {processing_steps}

FUNCTION SIGNATURE - USE EXACTLY THIS:
def visualize(data: List[NormalizedOpenMeteoData]) -> go.Figure:
    '''Process raw climate data from OpenMeteo API and generate a Plotly visualization.'''

DATA STRUCTURE:
NormalizedOpenMeteoData is a dataclass with:
- metadata: Optional[pd.DataFrame] = Field(description="Dataframe containing data unrelated to time resolution")
- hourly_data: Optional[pd.DataFrame] = Field(description="Dataframe with hourly data")
- daily_data: Optional[pd.DataFrame] = Field(description="Dataframe with daily data")

Data preview: {data_preview}

ALLOWED LIBRARIES - STRICTLY ONLY THESE:
- pandas (as pd)
- numpy (as np) 
- All Plotly libraries (plotly.graph_objects as go, plotly.express as px, plotly.subplots...)
- typing (for List, Optional)
- Python standard libraries

REQUIREMENTS:
1. Use Plotly to create the visualization
2. Validate input structure
3. Optimize for large datasets
4. Clear axes labels and title
5. Include legend for multiple traces
6. Avoid cluttering visualization

CRITICAL OUTPUT RULES:
1. ONLY output the complete visualize() function
2. NO code block markers or surrounding text
3. NO scipy, matplotlib, seaborn, or other external libraries
4. Function must be self-contained, with all necessary imports inside the function
5. Keep the code simple and clear. Avoid unnecessary complexity.
"""

########################
## Explanation Prompts
########################

EXPLANATION_PLAN_PROMPT = """

Given a climate visualization, create a detailed explanation plan to help the user better understand the visualization so he can make more sound decisions.

You should tailor the plan to fit the user persona.
{persona}

The scenario that was given to the user is the following : 
{scenario}

He had to allocate budget in these options :
{options}

The goal of the explanation is to make the user more aware of the climate change trends and patterns, and how they can relate to the visualization.

Please provide a structured outline addressing important elements of the visualization to help guide the user into making decisions.
"""


EXPLANATION_GENERATION_PROMPT = """
Based on the explanation plan provided, generate a comprehensive yet accessible explanation of the climate visualization. Your explanation should:

{explanation_plan}

Here's information about the data that has been used
{data_description}

Ensure your explanation is clear, short and engaging.
"""

SCENARIO_GENERATION_PROMPT = """
Create a realistic civic decision-making scenario for {location}. The scenario should:
1. Present a specific local government or community decision with a budget of approximately that seems realistic for the location (Big city would be around 1M$, small town around 100k$)
2. Offer 4 distinct policy or infrastructure options that adress issues in the city. (Education, Infrastructure, etc.) They should be really short and straightforward (few words)
3. Present one option that MUST be linked to the topic of {climate_topic}
4. Include local context and locations into the scenario and problem that may be known to citizens (e.g., recent extreme weather events, bad road infrastructure, etc.)
5. Be written in a neutral tone that doesn't favor any particular option
6. Include a brief explanation of why this decision is relevant to local residents

Focus on making the scenario believable, locally relevant, and engaging for the user. In the scenario description, all options should be equally introduced and with sufficient information to make an informed decision.
Do not make the scenario centered around climate change. It should be a general civic decision-making scenario with ONLY one option related to climate change.
"""


###########################
## Complexity Level Prompts
###########################

LVL0_EXP_PROMPT = """
You are interacting with a user who is new to climate science, environmental studies, and data visualization. Your responses should:
- Use simple, non-technical language that is easy to understand.
- Provide clear, basic explanations of climate concepts and visualizations.
- Break down complex ideas into relatable analogies or examples (e.g., "CO2 is like a blanket that traps heat around the Earth").
- Avoid scientific jargon. If technical terms are necessary, explain them in simple terms.
- Highlight the significance of the visualization in a way that connects to everyday life.
- Focus on building curiosity and foundational understanding.
- Use a friendly, supportive tone to make the user feel comfortable asking questions.
- Encourage exploration and learning without overwhelming the user.
"""

LVL1_EXP_PROMPT = """
You are communicating with a user who has a basic understanding of climate science, environmental concepts, and data visualization. Your responses should:
- Use technical terms when appropriate, but always provide clear explanations.
- Offer moderate-depth analysis of visualizations, explaining trends and patterns in the data.
- Discuss the broader implications of climate data (e.g., how rising temperatures affect weather patterns or ecosystems).
- Include statistical context where relevant, but keep it accessible (e.g., "Temperatures have risen by 1°C over the past century, which may not sound like much, but has significant impact as per IPCC scientists").
- Draw connections between different environmental indicators (e.g., how CO2 levels relate to ocean temperatures).
- Encourage critical thinking and deeper exploration of the topic.
- Use a professional yet engaging tone that balances clarity with depth.
"""

LVL2_EXP_PROMPT = """
You are engaging with a user who has a strong understanding of climate science, environmental concepts, and data visualization. Your responses should:
- Use precise, domain-specific language when necessary, but ensure clarity and relevance.
- Provide in-depth, technical analysis of data and visualizations, including trends, anomalies, and uncertainties.
- Discuss complex interdependencies in environmental systems (e.g., feedback loops between Arctic ice melt and global warming).
- Offer sophisticated statistical interpretations and insights (e.g., confidence intervals, predictive modeling).
- Highlight nuanced insights and encourage the user to think critically about the data.
- Use a scholarly yet approachable tone that assumes prior knowledge but remains accessible.
- Welcome advanced technical discussions and provide opportunities for deeper inquiry.
"""


LVL0_VIZ_PROMPT = """
You are generating visualizations for users new to data visualization and climate science. Your visualization outputs should:
- Use simple chart types (bar charts, line graphs, pie charts)
- Limit data points and variables shown simultaneously
- Include clear, prominent titles and labels
- Use intuitive color schemes (e.g., blue=cold, red=hot)
- Include basic legends and annotations
- Use rounded numbers and simplified scales
- Ensure all text is easily readable at standard viewing sizes
"""

LVL1_VIZ_PROMPT = """
You are generating visualizations for users with basic visualization literacy. Your outputs may include one or more of the following if relevant:
- Utilize intermediate chart types (scatter plots, box plots, stacked charts...)
- Layer 2-3 related variables in a single visualization
- Include statistical annotations where relevant (trend lines, confidence intervals)
- Use color schemes optimized for data type (sequential, diverging, categorical)
- Add detailed axis labels with units
- Enable basic comparative analysis
"""

LVL2_VIZ_PROMPT = """
You are generating visualizations for users with moderate literacy in data visualization. Your outputs may include one or more of the following if relevant:
- Employ advanced visualization types (heat maps, bubble charts...)
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

OUTPUT_LANGUAGE_PROMPT = """
For your answer, whether it's code, text or a visualization provide all the text that will be shown to the user in {lang}
This includes any labels, titles, descriptions, or explanations that will be directly visible to the user.

If your output is code, it should follow conventions and be written using English.
"""

ANTHROPIC_SYSTEM_PROMPT = """
You are a climate visualization expert and teacher. You are part of a process to generate a climate visualization with other experts.
Your role is to adapt to your audience knowledge level and provide clear visualization and explanations to help understand how climate change affects their environment.
"""

ANTHROPIC_STRUCTURED_OUTPUT_PROMPT = """
Your response must be ONLY valid JSON that EXACTLY matches this Python type:
{response_format}

Do not include any explanations, only the JSON itself. No markdown, no code blocks, no additional text.
"""