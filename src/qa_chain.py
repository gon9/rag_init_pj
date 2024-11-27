from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI


def create_qa_chain(vectorstore, openai_api_key):
    # ChatOpenAIモデルを初期化
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", #"map_reduce"
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}) # default 4
    )
    return qa_chain

def answer_question(qa_chain, question):
    result = qa_chain({"query": question})
    return result["result"]