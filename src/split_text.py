from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_text(text)
