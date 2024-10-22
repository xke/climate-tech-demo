import streamlit as st
import agentops
import os 

from crewai import Crew, Process, Agent, Task

#from langchain_community.llms import HuggingFaceHub
#from langsmith import traceable
import weave

#from agents.news_analysis_agent import get_news_analysis_agent
#from agents.sec_filings_agent import get_sec_filings_agent
#from agents.technical_indicators_agent import get_technical_indicators_agent

from agents.ira_agent import get_ira_agent
from agents.eli_agent import get_eli_agent

from textwrap import dedent
import os
from utils import CustomHandler, get_llm

from dotenv import load_dotenv
load_dotenv()

from datetime import date

#@weave.op()
#@agentops.record_function('log_run')
#def log_run(state, model, company, historical_horizon_in_years, prediction_time_horizon_in_years,
#             news_analysis_agent_enabled, sec_filings_agent_enabled, technical_indicators_agent_enabled, result):     
#    return result

#@weave.op()
#agentops.record_function("run_crew")
def run_crew(model, user_input, zip_code, property_type, ira_agent_enabled, ira_agent, eli_agent_enabled, eli_agent):

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Define tasks to run, and the agent to run it with

    tasksList = []
    agentsList = []

    if ira_agent_enabled:
        agentsList.append(ira_agent)
        analyze_ira_task = Task(
            description=dedent(f"""
                Please respond to this user inquiry in a helpful and friendly way:
                {user_input}
            """),
            expected_output=dedent(f"""
                Your output is a response to the user inquiry.
                Where possible, refer to the name of the source file, and
                the page number where information can be found.
            """),
            agent=ira_agent
        )
        tasksList.append(analyze_ira_task)


    if eli_agent_enabled:
        agentsList.append(eli_agent)
        analyze_eli_task = Task(
            description=dedent(f"""
                Please respond to this user inquiry in a helpful and friendly way:
                {user_input}
                Here is the zip code: {zip_code}  
                Here is the property type: {property_type}  
            """),
            expected_output=dedent(f"""
                Your output is a response to the user inquiry
                as returned from the ELI API.
            """),
            agent=eli_agent
        )
        tasksList.append(analyze_eli_task)

    #print(agentsList)
    #print(tasksList)

    if len(agentsList)==0:
        return "No agents found. Please choose at least one agent."
    
    # Set up the crew and process tasks hierarchically
    project_crew = Crew(
        tasks=tasksList,
        agents=agentsList,
        process=Process.sequential,
        manager_llm=get_llm(model), # only required for hierarchical process
        manager_callbacks=[CustomHandler("Manager")]
    )

    report = project_crew.kickoff()
    return report


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":

    st.set_page_config(page_title="Climate-Tech Demo", page_icon="🍀", layout="wide")

    st.subheader("🍀 Understanding renewable energy incentives",
                 divider="green", anchor=False)
    
    # Set up the Streamlit UI customization sidebar
    st.sidebar.title('Customizations')

    model = st.sidebar.selectbox(
       'Choose AI model to use',
       ['gpt-3.5-turbo', 'gpt-4o', 'claude-3-5-sonnet-20240620', 'claude-3-haiku-20240307', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
       index=0, # default index
    )

    zip_code = st.sidebar.text_input(
        'Zip code',
        placeholder="94109"
    )

    property_type = st.sidebar.selectbox(
        'Property type',
        ['single_family', 'multifamily', 'commercial'],
        index=0, # default index
    )

    

    # Define agents with their specific roles and goals

    st.sidebar.write("")
    st.sidebar.write("Choose CrewAI agent(s) to use:")

    #news_analysis_agent_enabled = st.sidebar.checkbox(
    #    'News analysis agent',
    #    value=True
    #)

    #if news_analysis_agent_enabled:
    #    news_analysis_agent = get_news_analysis_agent(chosen_llm)

    ira_agent_enabled = st.sidebar.checkbox(
        'Inflation Reduction Act agent (RAG PDF search)',
        value=True
    )

    ira_agent = None

    if ira_agent_enabled:
        ira_agent = get_ira_agent(model)

    eli_agent_enabled = st.sidebar.checkbox(
        'ELI agent (via ELI API)',
        value=True
    )

    eli_agent = None

    if eli_agent_enabled:

        if zip_code == None:
            zip_code = "94109"
        if property_type == None:
            property_type = "single_family"

        eli_agent = get_eli_agent(model, zip_code, property_type)


    st.sidebar.write("")
    st.sidebar.write("")
   # st.sidebar.markdown(
   # """
   # *This is an [open-source demo app](https://github.com/xke/). Use the AI output at your own risk.*
   # 
   # """,
   #     unsafe_allow_html=True
   # )
  
    
    # Initialize the message log in session state if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "##### What do you want to know about?"}]


    # Display existing messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input

    if chat_input := st.chat_input():
        #agentops.init(tags=["jazzmine-entropy", chat_input, model])

        #agentops.record(agentops.ActionEvent([chat_input, model]))

        report = run_crew(model, chat_input, zip_code, property_type, ira_agent_enabled, ira_agent, eli_agent_enabled, eli_agent)

        # Display the final result
        result = f"##### Manager's Final Report: \n\n {report}"
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)

        #if result: 
        #    agentops.end_session('Success')
        #else:
        #    agentops.end_session('Failure')