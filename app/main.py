from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from .models import ChatDescriptionRequest, ChatVisualizationRequest, ChatVisualizationResponse, ScenarioRequest, ScenarioResponse, PersonaRequest, ChatRequest
from dotenv import load_dotenv
from time import sleep
import os
load_dotenv()

from .process import set_complexity_level, generate_visualization, get_complexity_level_prompts

from .ai import anthropic_client
from .constants import USER, DEVELOPER, AVAILABLE_SCENARIOS

from .prompts import EXPLANATION_PLAN_PROMPT, EXPLANATION_GENERATION_PROMPT, SCENARIO_GENERATION_PROMPT, SCENARIO_EXPLANATION

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_END_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/scenario")
async def get_scenario(request: Request, body: ScenarioRequest) -> ScenarioResponse:
    lang = request.headers.get('Accept-Language')
    print(lang)

    scenario = anthropic_client.structured_completion(
        messages=[
            {"role": USER, "content": SCENARIO_GENERATION_PROMPT.format(
                climate_topic=body.topic, 
                location=body.location
            )},],
        response_format=ScenarioResponse,
        system_prompt="You are a policy maker in {body.location} and you have to create a realistic scenario to assess citizen's decision making in public budget spending / allocation. Depending on the language, adapt the currency for the budget. (e.g. JPY for Japanese, USD for English)",
        lang=lang
    )


    return scenario


@app.post("/chat/persona")
async def get_persona(request: Request, body: PersonaRequest):
    complexity_level = set_complexity_level(f"{body.description}. My age group is {body.age_group}")
    return {"complexity_level": complexity_level}


@app.post("/chat/visualization")
async def visualize(request: Request, body: ChatVisualizationRequest) -> ChatVisualizationResponse:
    lang = request.headers.get('Accept-Language')
    try:
        fig = generate_visualization(
            body.messages,
            body.complexity_level,
            body.user_description,
            body.location,
            body.chat_id,
            body.scenario,
            body.topic,
            body.options,
            lang
        )
        return ChatVisualizationResponse(visualization=fig)
    except Exception as e:
        print(e)
        return HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/chat/description")
async def describe(request: Request, body: ChatDescriptionRequest):
    """
    API route for generating visualization descriptions with streaming response.
    
    Args:
        request (ChatDescriptionRequest): The request containing chat_id and image
        
    Returns:
        StreamingResponse: A streaming response with the visualization description
    """
    # Get data from file or use empty string if file not found
    try:
        with open(f"{body.chat_id}.txt", "r") as file:
            data_description = file.read()
    except FileNotFoundError:
        data_description = ""
    
    lang = request.headers.get('Accept-Language')

    _, description_complexity = get_complexity_level_prompts(body.complexity_level)
    
    # Get explanation plan
    explanation_plan = anthropic_client.completion(
        messages=[
            {"role": DEVELOPER, "content": description_complexity},
            {"role": USER, "content": [
                {
                    "type": "text",
                    "text": SCENARIO_EXPLANATION.format(scenario=body.scenario, options=body.options),
                },
                {
                    "type": "text",
                    "text": EXPLANATION_PLAN_PROMPT,
                },
                {
                    "type": "image",
                    "source": {"type": "base64",
                              "data": body.image,
                              "media_type": "image/png"},
                },
            ]},
        ],
        temperature=0.7,
        max_tokens=300,
        lang=lang
    )
    
    # Setup messages for explanation generation
    messages = [
        {"role": DEVELOPER, "content": description_complexity},
        {"role": USER, "content": [
            {
                    "type": "text",
                    "text": SCENARIO_EXPLANATION.format(scenario=body.scenario, options=body.options),
                },
            {
                "type": "text",
                "text": EXPLANATION_GENERATION_PROMPT.format(
                    explanation_plan=explanation_plan,
                    data_description=data_description
                ),
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": body.image,
                    "media_type": "image/png"
                },
            },
        ]},
    ]

    return StreamingResponse(
        content=anthropic_client.streaming(
            messages=messages,
            temperature=0.7,
            max_tokens=300,
            lang=lang
        ),
        media_type="text/event-stream"
    )


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    """
    API route for generating chat responses with streaming response.
    Args:
    request (Request): The request containing the chat_id and message
    Returns:
    StreamingResponse: A streaming response with the chat responses
    """
    lang = request.headers.get('Accept-Language', 'en')

    messages = anthropic_client._convert_to_anthropic_format(body.messages)

    generator = anthropic_client.streaming(
        messages=messages,
        temperature=0.7,
        max_tokens=300,
        lang=lang)
    
    return StreamingResponse(
        generator,
        media_type="text/event-stream"
    )




@app.get("/test/")
async def test() -> ChatVisualizationResponse:
    sleep(2)
    viz = '{"data":[{"line":{"color":"red","width":2},"mode":"lines","name":"Average Summer Temperature","x":[1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[23.840372670807454,23.540372670807454,23.668478260869566,23.36863354037267,24.290372670807454,23.88571428571429,23.43944099378882,23.807608695652174,24.482453416149067,24.02639751552795,24.18726708074534,24.07034161490683,23.739906832298136,23.429192546583852,24.4332298136646,24.122360248447205,23.298757763975157,24.22701863354037,24.27003105590062,24.5166149068323,24.63121118012422,24.039906832298133,24.791459627329193,24.204037267080746,24.595341614906832,24.55667701863354,24.477950310559006,25.057298136645965,24.167080745341615,24.728105590062114,24.645341614906833,24.44347826086957,24.700931677018637,24.32577639751553,24.440993788819874,24.298291925465836,24.467391304347824,24.60512422360248,24.606521739130436,24.486024844720497,24.752484472049687,24.70388198757764,24.367857142857144,24.456211180124225],"type":"scatter"}],"layout":{"template":{"data":{"barpolar":[{"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"bar":[{"error_x":{"color":"rgb(36,36,36)"},"error_y":{"color":"rgb(36,36,36)"},"marker":{"line":{"color":"white","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"carpet":[{"aaxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"baxis":{"endlinecolor":"rgb(36,36,36)","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"rgb(36,36,36)"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"choropleth"}],"contourcarpet":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"contourcarpet"}],"contour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"contour"}],"heatmapgl":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmapgl"}],"heatmap":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"heatmap"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2dcontour"}],"histogram2d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"histogram2d"}],"histogram":[{"marker":{"line":{"color":"white","width":0.6}},"type":"histogram"}],"mesh3d":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scattermapbox"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolargl"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterpolar"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"rgb(237,237,237)"},"line":{"color":"white"}},"header":{"fill":{"color":"rgb(217,217,217)"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":1,"tickcolor":"rgb(36,36,36)","ticks":"outside"}},"colorscale":{"diverging":[[0.0,"rgb(103,0,31)"],[0.1,"rgb(178,24,43)"],[0.2,"rgb(214,96,77)"],[0.3,"rgb(244,165,130)"],[0.4,"rgb(253,219,199)"],[0.5,"rgb(247,247,247)"],[0.6,"rgb(209,229,240)"],[0.7,"rgb(146,197,222)"],[0.8,"rgb(67,147,195)"],[0.9,"rgb(33,102,172)"],[1.0,"rgb(5,48,97)"]],"sequential":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]],"sequentialminus":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"colorway":["#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"],"font":{"color":"rgb(36,36,36)"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"white","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"white","polar":{"angularaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","radialaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"scene":{"xaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"zaxis":{"backgroundcolor":"white","gridcolor":"rgb(232,232,232)","gridwidth":2,"linecolor":"rgb(36,36,36)","showbackground":true,"showgrid":false,"showline":true,"ticks":"outside","zeroline":false,"zerolinecolor":"rgb(36,36,36)"}},"shapedefaults":{"fillcolor":"black","line":{"width":0},"opacity":0.3},"ternary":{"aaxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"baxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"},"bgcolor":"white","caxis":{"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside"}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"},"yaxis":{"automargin":true,"gridcolor":"rgb(232,232,232)","linecolor":"rgb(36,36,36)","showgrid":false,"showline":true,"ticks":"outside","title":{"standoff":15},"zeroline":false,"zerolinecolor":"rgb(36,36,36)"}}},"shapes":[{"line":{"color":"gray","dash":"dash"},"type":"line","x0":0,"x1":1,"xref":"x domain","y0":24.25440782044043,"y1":24.25440782044043,"yref":"y"}],"annotations":[{"showarrow":false,"text":"Historical Average: 24.3°C","x":1,"xanchor":"right","xref":"x domain","y":24.25440782044043,"yanchor":"top","yref":"y"}],"title":{"font":{"size":16},"text":"Nagoya Summer Temperature Trends (1980-2023)","x":0.5,"xanchor":"center"},"xaxis":{"tickfont":{"size":12},"title":{"text":"Year","font":{"size":14}},"tickmode":"linear","dtick":5,"showgrid":true,"gridwidth":1,"gridcolor":"#E5E5E5","zeroline":true,"zerolinewidth":1,"zerolinecolor":"#808080"},"yaxis":{"tickfont":{"size":12},"title":{"text":"Temperature (°C)","font":{"size":14}},"showgrid":true,"gridwidth":1,"gridcolor":"#E5E5E5","zeroline":true,"zerolinewidth":1,"zerolinecolor":"#808080"},"legend":{"font":{"size":12},"yanchor":"top","y":0.99,"xanchor":"left","x":0.01},"margin":{"t":80,"l":50,"r":50,"b":50},"showlegend":true,"plot_bgcolor":"white","paper_bgcolor":"white","font":{"size":12},"autosize":true}}'
    return ChatVisualizationResponse(visualization=viz)

