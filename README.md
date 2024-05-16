# Summarizer Assistant

This repo features some of my experimentation with the fundamental LLM task of summarization. 
I utilize the OpenAI API and Langchain to build a series of summarizer assistants. The assistants 
can interact with the web, providing concise bullet point summaries of a page at a 
given URL. They can also summarize .txt files and free-form string inputs. I use techniques like OpenAI
function calling to allow the assistant to infer the input type and retrieve the necessary arguments.
