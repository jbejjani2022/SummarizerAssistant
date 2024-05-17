from langchain_community.document_loaders import WebBaseLoader

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