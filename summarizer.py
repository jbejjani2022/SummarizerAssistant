# Usage: python summarizer.py -t/--type {'n', 'm', 'c'} text
# ex. python summarizer.py -t m https://en.wikipedia.org/wiki/Ludwig_van_Beethoven
# python summarizer.py -t n data/romeojuliet.txt
# where text is a string, .txt file, or URL
# 'n' specifies naive summarization strategy, 'm' specifies map_reduce strategy

import naive_summarizer
import map_reduce_summarizer
import chunk_summarizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='summarizer',
                    description='Summarize an input string, the contents of a .txt file, or a web page given its URL')
    
    parser.add_argument('-t', '--type', required=True, choices=['n', 'm', 'c'],
                        help='specify the summarization type: -n for naive truncation, -m for map_reduce, -c for chunking')
    parser.add_argument('text', help='a string, .txt file, or URL to be summarized')
    args = parser.parse_args()

    text = args.text
    if args.type == 'n':
        summary = naive_summarizer.summarize(text)
    elif args.type == 'm':
        summary = map_reduce_summarizer.summarize(text)
    elif args.type == 'c':
        summary = chunk_summarizer.summarize(text)
    print(summary)