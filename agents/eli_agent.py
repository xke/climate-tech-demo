from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from sec_api import QueryApi
from unstructured.partition.html import partition_html
from crewai import Agent

from utils import CustomHandler, get_llm
from langchain.tools import tool
import requests
import os 

#import weave

#@weave.op()

@tool("Get ELI API indicators")
def get_eli_tool(zip_code, property_type):
    """
    Get data from the ELI API. 
    The input is the zip code and property type
    """

    programs_url = "https://api.eli.build/programs"

    payload = {
        "address": {
            "zipcode": zip_code
        },
        "property_type": property_type,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer " + os.environ['ELI_AUTH_TOKEN']
    }

    programs_response = requests.post(programs_url, json=payload, headers=headers)

    incentives_url = "https://api.eli.build/incentives"

    payload = {
        "address": {
            "zipcode": zip_code
        },
        "property_type": property_type,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }

    incentives_response = requests.post(incentives_url, json=payload, headers=headers)


    return str(programs_response) + " " + str(incentives_response)



def get_eli_agent(model, zip_code, property_type):
  
  #eli_tool = get_eli_tool(zip_code, property_type)

  return Agent(
          role='The More Accurate ELI API connector',
          goal="""Being the best at gathering and interpreting data from the ELI API

                Here's the zip code: {zip_code}
                Here is the property type: {property_type}
          
                """,
          backstory="""Takes a zip code and property type, and uses the ELI tool to get relevant info""",
          verbose=True,
          allow_delegation=False,
          tools=[
              get_eli_tool,
          ],
          llm=get_llm(model),
          callbacks=[CustomHandler("ELI Agent")]
      )
