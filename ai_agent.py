import requests
from langchain import hub
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Step 1: Define the required tools
search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherapi.com/v1/current.json?key=<put-your-api-key-before-use>&q={city}&aqi=no"

    response = requests.get(url)

    return response.json()


# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=gemini, tools=[search_tool, get_weather_data], prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)


# Step 5: Invoke
response = agent_executor.invoke(
    {
        "input": "Find the capital of Odisha (state of india), then find it's current weather condition"
    }
)

print(response)
print(response["output"])
