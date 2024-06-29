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
from crewai_tools import PDFSearchTool
#import weave

#@weave.op()
def get_ira_agent(model):

  ira_tool = PDFSearchTool(
      pdf='agents/pdf/Inflation-Reduction-Act-Guidebook.pdf',
      config=dict(
          llm=dict(
              provider="openai", # or google, openai, anthropic, llama2, ...
              config=dict(
                  model=model,
                  temperature=0.1,
                  # top_p=1,
                  stream=True,
              ),
          ),
          #embedder=dict(
          #    provider="openai", # or openai, ollama, ...
              #config=dict(
                  #model="models/embedding-001", # just use defeault
                  #task_type="retrieval_document",
                  # title="Embeddings",
              #),
          #),
      )
  )

  return Agent(
          role='The More Insightful Inflation Reduction Act Researcher and Analyst',
          goal="""Being the best at gathering and interpreting Inflation Reduction Act (IRA) data""",
          backstory="""The most seasoned and experienced Inflation Reduction Act (IRA) researcher and 
          analyst with lots of expertise in understanding which information is most relevant
          user inquiries.""",
          verbose=True,
          allow_delegation=False,
          tools=[
              ira_tool,
          ],
          llm=get_llm(model),
          callbacks=[CustomHandler("IRA Agent")]
      )
