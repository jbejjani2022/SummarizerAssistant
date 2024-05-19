# Summarizer Assistant

This repo features some of my experimentation with the fundamental LLM task of summarization. 
I utilize the OpenAI API and Langchain to build a series of summarizer assistants. The assistants 
can interact with the web, providing concise bullet point summaries of a page at a 
given URL. They can also summarize .txt files and free-form string inputs. I use techniques like OpenAI
function calling to allow the assistant to infer the input type and retrieve the necessary arguments. The main constraint is the token limit/context window size of the LLM used, since this limits how much text you can summarize per call to the LLM. The challenge is then to strike a balance across accuracy, comprehensiveness, computational efficiency, and cost.

I implement 3 summarization strategies:
1. naive truncation: Truncate texts that exceed the `token_limit` defined in `settings.py`, then summarize.
2. map reduction: Utilize Langchain's map reduce to break the text into chunks that each fit within the `token_limit`, then summarize the chunks and summarize the summaries to produce a final output.
3. chunking: If the text exceeds the `token_limit`, break it into chunks that fit within the limit, summarize each chunk, then patch together the summaries to make one large summary.

Usage:

Summarize a text by calling
`python summarizer.py -t/--type {'n', 'm', 'c'} text`
Where `text` is a string, .txt file, or URL, and `n` indicates naive truncation, `m` is map reduction, and `c` is chunking.
Ex.
`python summarizer.py -t m https://en.wikipedia.org/wiki/Ludwig_van_Beethoven`
`python summarizer.py -t n data/romeojuliet.txt`

Here are some of my thoughts about each strategy after experimenting:

Naive truncation is the most computationally efficient because it requires one call to the LLM, but it loses all information about text that is truncated. Map reduction produces a concise final summary that covers the entire text, but for longer texts it generally yields a coarse-grained summary that does not include much detail. Chunking provides the longest and most detailed final summary, but is unable to capture any contextual/semantic dependencies across chunks.

Next steps:
- Use K-Means vector clustering for very large documents. The aim is to group chunks into 'meaning clusters' and sample a representative chunk from each cluster to make a comprehensive final summary that covers all topics in the document that are both important and distinct.
