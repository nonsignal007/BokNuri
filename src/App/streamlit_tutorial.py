import streamlit as st
import tiktoken
## tokenizer 하는 것
from loguru import logger

from langchain.chains import ConversationalRetrievalChain ## 메모리를 가지고 있음
from langchain.chat_models import ChatOpenAI ## openai 모델

## 여러개의 파일들이 들어가도 읽어오도록 하느것
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

## spliter
from langchain.text_splitter import RecursiveCharacterTextSplitter
## embedding
from langchain.embeddings import HuggingFaceEmbeddings

## langchain 의 메모리를 몇개까지 넣을것인지를 저장
from langchain.memory import ConversationBufferMemory
## 임시 벡터 제현
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:") ## 아이콘 설정 : 미리 설정한 것만 사용할 수있음

    st.title("_Private Data :red[QA Chat]_ :books:")
 
    ## 미리 session_state 변수를 정의한 것
    if "conversation" not in st.session_state:
        st.session_state.conversation = None 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    ## 하위 구조로 만들어서 사용할 수 있음
    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")

    ## 만약에 process 버튼을 누르면 실행
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."): ## 질문창을 만들어줌
        st.session_state.messages.append({"role": "user", "content": query})

        ## 사용장의 질문 표시
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): ## 로딩중을 표시
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    ## tokenizer를 사용하여 길이를 계산
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    # 업로드한 파일을 읽어서 텍스트로 변환
    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len ## tokenizer를 사용하여 길이를 계산
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()