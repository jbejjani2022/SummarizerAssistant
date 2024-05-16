# test functionality of summarizer.py

from summarizer import get_page_content

# some sample URLs to try
# https://en.wikipedia.org/wiki/Ludwig_van_Beethoven
# https://www.nytimes.com/2024/05/15/us/politics/trump-biden-debate-june.html

url = "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven"
data = get_page_content(url)

print(len(data))
print(data)