from utilities import get_txt_content
from settings import model, token_limit
import sys
import tiktoken
import spacy
from spacy.lang.en import English
import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from utilities import get_page_content, get_txt_content, function_descriptions

file = "data/romeojuliet.txt"
text = get_txt_content(file)

 
def text_to_chunks(text):
    # chunk the text using spaCy sentence splitter
    # utilizes a tokenizer to prevent splitting up sentences during chunking
    nlp = spacy.load("en_core_web_sm")
    sentences = nlp(text).sents
    
    # for i, sent in enumerate(list(sentences)[:5]):
    #   print(f"sentence {i}: ", sent.text)
    
    encoding = tiktoken.encoding_for_model(model)
    chunks = [[]]
    chunk_total_tokens = 0
    
    # iterate over each sentence, adding the sentence to current chunk
    # and creating new chunk whenever current chunk hits token limit
    for s in sentences:
        encoded_s = encoding.encode(s.text)
        num_tokens = len(encoded_s)
        if chunk_total_tokens + num_tokens > token_limit:
            # start new chunk
            chunks.append([])
            chunk_total_tokens = num_tokens
        else:
            chunk_total_tokens += num_tokens 
        
        chunks[-1].append(s.text)
        
    return chunks


def summarize_chunk(chunk, client, prompt, output):
    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "user",
             "content": prompt},
            {"role": "function",
             "name": output.function_call.name,
             "content": chunk}
        ],
        functions = function_descriptions,
        function_call = "auto"
    )

    summary = completion.choices[0].message
    return summary.content


def summarize(text):
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY']
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
    
    chunks = text_to_chunks(text)
    chunk_summaries = []
    for chunk in chunks:
        chunk_summary = summarize_chunk(" ".join(chunk), client, prompt, output)
        chunk_summaries.append(chunk_summary)
        
    return " ".join(chunk_summaries)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chunk_summarizer.py <string, .txt file, or url>")
        sys.exit(1)
    text = sys.argv[1]
    
    summary = summarize(text)
    print(summary)