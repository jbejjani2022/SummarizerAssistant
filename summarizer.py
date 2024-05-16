# Usage: python summarizer.py --type {'n', 'mr'} text
# where text is a string, .txt file, or URL
# 'n' specifies naive summarization strategy, 'mr' specifies map_reduce strategy

import naive_summarizer
import map_reduce_summarizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='summarizer',
                    description='Summarize an input string, the contents of a .txt file, or a web page given its URL')
    
    parser.add_argument('-t', '--type', required=True, choices=['n', 'mr'],
                        help='specify the summarization type: -n for naive truncation, -mr for map_reduce')
    parser.add_argument('text', help='a string, .txt file, or URL to be summarized')
    args = parser.parse_args()

    text = args.text
    if args.type == 'n':
        summary = naive_summarizer.summarize(text)
    elif args.type == 'mr':
        summary = map_reduce_summarizer.summarize(text)
    print(summary)