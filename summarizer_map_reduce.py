from dotenv import load_dotenv
import os

from summarizer_naive import get_txt_content, get_page_content

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

load_dotenv()
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ['OPENAI_API_KEY'])

file = "data/romeojuliet.txt"
text = get_txt_content(file)

# split the text into chunks that fit into the prompt limit
# each token is about 4 characters, so 10000 character chunks are ~2500 tokens each
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
docs = text_splitter.create_documents([text])

num_docs = len(docs)
# print(docs)

num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)

# print (f"We have {num_docs} documents.\nThe first one has {num_tokens_first_doc} tokens.")

# summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')
# summary = summary_chain.run(docs)
# print(summary)

map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
# map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = """
    Summarize the following text delimited by triple backquotes.
    Format your response in bullet points that cover the key points of the text.
    ```{text}```
    BULLET POINT SUMMARY:
    """
# combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

# summary_chain = load_summarize_chain(llm=llm,
#                                      chain_type='map_reduce',
#                                      map_prompt=map_prompt_template,
#                                      combine_prompt=combine_prompt_template)
# 
# summary = summary_chain.run(docs)
# print(summary)

@tool
def get_url_content(url: str) -> str:
    """Useful for getting the contents of the web page at the url."""
    return get_page_content(url)

@tool
def get_file_content(file: str) -> str:
    """Useful for getting the contents of a .txt file."""
    return get_txt_content(file)

tools = [get_url_content, get_file_content]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a very powerful assistant great at summarizing a variety of text inputs",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools)
prompt = "Summarize the contents of https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/ in bullet point format."

# summary = list(agent_executor.stream({"input": "Summarize the contents of https://python.langchain.com/v0.1/docs/modules/agents/how_to/custom_agent/ in bullet point format."}))
summary = agent_executor.invoke({"input": prompt})
print(summary['output'])