import torch
import streamlit as st
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.memory import StreamlitChatMessageHistory
import tiktoken

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
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
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "upskyy/bge-m3-korean"
    model_kwargs = {'device': DEVICE ,'cache_dir': 'weights'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_llm():
    model_id = "davidkim205/Ko-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="weights")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="weights"
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        device=DEVICE
    )
    
    return HuggingFacePipeline(pipeline=pipe)

def get_conversation_chain(vectorstore):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        ),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def main():
    st.set_page_config(
        page_title="EyE Chat",
        page_icon=":eyes:"
    )
    
    st.title("_Private Data :red[QA Chat]_ :eyes:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload your file",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )
        process = st.button("Process")
    
    if process:
        if not uploaded_files:
            st.error("Please upload at least one file.")
            return
            
        with st.spinner("Processing files..."):
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.processComplete = True
            st.success("Files processed successfully!")
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"
        }]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input("질문을 입력해주세요."):
        if not st.session_state.processComplete:
            st.error("Please process some documents first!")
            return
            
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = st.session_state.conversation
                result = chain({"question": query})
                response = result['answer']
                source_documents = result['source_documents']
                
                st.markdown(response)
                
                if source_documents:
                    with st.expander("참고 문서 확인"):
                        for idx, doc in enumerate(source_documents):
                            st.markdown(f"문서 {idx + 1}: {doc.metadata['source']}")
                            with st.expander(f"문서 {idx + 1} 내용"):
                                st.markdown(doc.page_content)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

if __name__ == '__main__':
    main()