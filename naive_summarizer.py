# Summarize text passed as a command line argument
# Usage: python naive_summarizer.py <string, .txt file, or url>
# Uses 'stuffing' approach: summarizes entire text at once, but 
# truncates the text if it exceeds max token length

import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from utilities import get_page_content, get_txt_content, function_descriptions
from settings import model, token_limit
import tiktoken


def truncate_tokens(text, encoding_name, max_length=token_limit):
    """Truncate a text string based on max number of tokens"""
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded_text = encoding.encode(text)
    num_tokens = len(encoded_text)
    
    if num_tokens > max_length:
        print("The text was truncated.")
        text = encoding.decode(encoded_text[:max_length])
    return text


def summarize(text):
    # load OpenAPI API key from .env file
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
    )
    
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
    if len(sys.argv) != 2:
        print("Usage: python naive_summarizer.py <string, .txt file, or url>")
        sys.exit(1)
    text = sys.argv[1]
    summary = summarize(text)
    print(summary)