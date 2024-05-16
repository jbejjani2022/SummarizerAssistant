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
    