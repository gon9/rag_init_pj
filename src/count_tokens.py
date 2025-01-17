import tiktoken

def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    return len(tokens)
