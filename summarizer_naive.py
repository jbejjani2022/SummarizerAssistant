# Summarize text passed as a command line argument
# Usage: python summarizer.py <string, .txt file, or url>
# Performs naive truncation if input text exceeds max token length

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import WebBaseLoader
import tiktoken

# load OpenAPI API key from .env file

load_dotenv(find_dotenv())
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

function_descriptions = [
    {
        "name": "get_page_content",
        "description": "Get the contents of a web page given its URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the web page, e.g. https://en.wikipedia.org/wiki/OpenAI"
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "get_txt_content",
        "description": "Get the contents of a .txt file",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "The file name ending in .txt, e.g. shakespeare.txt"
                }
            },
            "required": ["file"],
        },
    }
]


def get_page_content(url):
    """Get the contents of the web page at the input url."""
    loader = WebBaseLoader(url)
    page = loader.load()
    return page[0].page_content


def get_txt_content(file):
    """Get the contents of the .txt file."""
    try:
        with open(file, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "The file was not found."
    except IOError:
        return "An error occurred while reading the file."


def truncate_tokens(text, encoding_name, max_length=2000):
    """Truncate a text string based on max number of tokens"""
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded_text = encoding.encode(text)
    num_tokens = len(encoded_text)
    
    if num_tokens > max_length:
        text = encoding.decode(encoded_text[:max_length])
    return text


def main():
    model = "gpt-3.5-turbo-0613"
    if len(sys.argv) != 2:
        print("Usage: python summarizer.py <string, .txt file, or url>")
        return 1
    
    text = sys.argv[1]
    
    prompt = f"""
    Summarize the following text delimited by triple backquotes. If it is a .txt file or URL,
    summarize the contents of the file or web page.
    Format your response in bullet points that cover the key points of the text.
    ```{text}```
    """
    
    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user",
             "content": prompt}
        ],
        functions = function_descriptions,
        function_call = "auto"
    )

    output = completion.choices[0].message
    
    if not output.function_call:
        return output.content
    
    # make the necessary function calls
    param = json.loads(output.function_call.arguments)
    f = eval(output.function_call.name)
    text = f(**param)
    text = truncate_tokens(text, model)
    
    second_completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user",
             "content": prompt},
            {"role": "function",
             "name": output.function_call.name,
             "content": text}
        ],
        functions = function_descriptions,
        function_call = "auto"
    )

    summary = second_completion.choices[0].message
    return summary.content
    
    
if __name__ == "__main__":
    summary = main()
    print(summary)