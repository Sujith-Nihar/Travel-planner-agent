
from crewai import Crew, Process
import streamlit as st



from crewai import Agent
#from TravelTools import search_web_tool, web_search_tool
from crewai import LLM
# from langchain_ollama.llms import OllamaLLM




from crewai import Task

def location_task(agent, from_city, destination_city, date_from, date_to):
    return Task(
        description=f"""
        Provide travel-related information including accommodations (list top 3 accomodation in {destination_city}), cost of living,
        visa requirements, transportation, weather(how is the weather {destination_city} from {date_from} to {date_to}), and local events.

        Traveling from: {from_city}
        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}

        Respond in FRENCH if the destination is in a French-speaking country.
        """,
        expected_output="A detailed markdown report with relevant travel data.",
        agent=agent,
        output_file='city_report.md',
    )

def guide_task(agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Provide a travel guide with attractions, food recommendations, and events.
        Tailor recommendations based on user interests: {interests}.
        
        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}
        """,
        expected_output="A markdown itinerary including attractions, food, and activities.",
        agent=agent,
        output_file='guide_report.md',
    )

def planner_task(context, agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Combine information into a well-structured itinerary. Include:
        - City introduction (4 paragraphs)
        - Daily travel plan with time allocations
        - Expenses and tips

        Destination: {destination_city}
        Interests: {interests}
        Arrival: {date_from}
        Departure: {date_to}
        """,
        expected_output="A structured markdown travel itinerary.",
        context=context,
        agent=agent,
        output_file='travel_plan.md',
    )


from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

# Web search tool using DuckDuckGo
@tool
def search_web_tool(query: str):
    """
    Searches the web and returns results.
    """
    search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
    return search_tool.run(query)

# Web scraping tool
#web_search_tool = WebsiteSearchTool()
#scrape_website_tool = ScrapeWebsiteTool()


# Initialize LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.5,
)


# Agents
guide_expert = Agent(
    role="City Local Guide Expert",
    goal="Provides information on things to do in the city based on user interests.",
    backstory="A local expert passionate about sharing city experiences.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=llm,
    allow_delegation=False,
)

location_expert = Agent(
    role="Travel Trip Expert",
    goal="Provides travel logistics and essential information.",
    backstory="A seasoned traveler who knows everything about different cities.",
    tools=[search_web_tool],  
    verbose=True,
    max_iter=5,
    llm=llm,
    allow_delegation=False,
)

planner_expert = Agent(
    role="Travel Planning Expert",
    goal="Compiles all gathered information to create a travel plan.",
    backstory="An expert in planning seamless travel itineraries.",
    tools=[search_web_tool],
    verbose=True,
    max_iter=5,
    llm=llm,
    allow_delegation=False,
)


# Streamlit App Title
st.title("🌍 AI-Powered Trip Planner")

st.markdown("""
💡 **Plan your next trip with AI!**  
Enter your travel details below, and our AI-powered travel assistant will create a personalized itinerary including:
 Best places to visit 🎡   Accommodation & budget planning 💰
 Local food recommendations 🍕   Transportation & visa details 🚆
""")

# User Inputs
from_city = st.text_input("🏡 From City", "Chicago")
destination_city = st.text_input("✈️ Destination City", "New York")
date_from = st.date_input("📅 Departure Date")
date_to = st.date_input("📅 Return Date")
interests = st.text_area("🎯 Your Interests (e.g., sightseeing, food, adventure)", "sightseeing and good food")

# Button to run CrewAI
if st.button("🚀 Generate Travel Plan"):
    if not from_city or not destination_city or not date_from or not date_to or not interests:
        st.error("⚠️ Please fill in all fields before generating your travel plan.")
    else:
        st.write("⏳ AI is preparing your personalized travel itinerary... Please wait.")

        # Initialize Tasks
        loc_task = location_task(location_expert, from_city, destination_city, date_from, date_to)
        guid_task = guide_task(guide_expert, destination_city, interests, date_from, date_to)
        plan_task = planner_task([loc_task, guid_task], planner_expert, destination_city, interests, date_from, date_to)

        # Define Crew
        crew = Crew(
            agents=[location_expert, guide_expert, planner_expert],
            tasks=[loc_task, guid_task, plan_task],
            process=Process.sequential,
            full_output=True,
            verbose=True,
        )

        # Run Crew AI
        result = crew.kickoff()

        # Display Results
        st.subheader("✅ Your AI-Powered Travel Plan")
        st.markdown(result)


        # Ensure result is a string
        travel_plan_text = str(result)  # ✅ Convert CrewOutput to string

        st.download_button(
            label="📥 Download Travel Plan",
            data=travel_plan_text,  # ✅ Now passing a valid string
            file_name=f"Travel_Plan_{destination_city}.txt",
            mime="text/plain"
        )
