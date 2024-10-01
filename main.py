from flask import Flask, jsonify, request, render_template
from langchain.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from uuid import uuid4
import os
import faiss
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

#Pre defined path and api key
index_path = 'faiss.index'
file_path = './data/'

#Change API key with yours in .env file
api_key = os.getenv("OPENAI_API_KEY")

# Loading and intializing predefined models 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
memory = ConversationBufferMemory() 
chat_model = ChatOpenAI(model="gpt-4o", temperature=0.7)


#Loading exist vector db if present else creating new
def load_vector():
    try:
        #loading existing vector db - faiss
        vector_store = faiss.read_index(index_path)
        retriever = vector_store.as_retriever()
        return vector_store, retriever
    except:
        # Creating new vector db
        index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        retriever = vector_store.as_retriever()
        return vector_store, retriever
# chatbot configurations intialising function to load depedencies.
def chatbot_init():
    vector_store, retriever = load_vector()
    prompt_template = """
    Use the following pieces of context to answer the question. If context is not enough use the information from indian goverment. 
    question contains history use memory also to get answer.
    {context}

    Question: {question}
    Give the answer in one line with repect to question within 50 words.
    Give answer like I'm 5
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Step 5: Create the LLM Chain using the chat model and the prompt template
    llm_chain = LLMChain(llm=chat_model, prompt=prompt)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context" )
    # Step 6: Create the Retrieval-based QA chain
    qa_chain = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    # Step 7: Define the tool that the agent will use
    tools = [
        Tool(
            name="Search with Vector DB",
            func=qa_chain.run,
            description="This tool searches the vector database to retrieve relevant information."
        )
    ]

    # Step 8: Initialize the agent
    agent = initialize_agent(
        tools,
        chat_model,
        agent_type="zero-shot-react-description",  # You can also experiment with different agent types
        handle_parsing_errors=True,
    )
    return agent



#document vector to vector the input file
def document_vectorizer(file_name, file_type):
    # Load the vector store and retriever from disk
    vector_store, retriever = load_vector()
    # Load and split the PDF document into manageable parts
    loader = PyPDFLoader(file_name)
    documents = loader.load_and_split()
    # Add metadata to each document
    for idx, doc in enumerate(documents):
        doc.metadata = {'category': file_type, 'page': idx + 1}
    # Generate unique IDs for the documents
    uuids = [str(uuid4()) for _ in range(len(documents))]
    # Check if the vector store is ready to accept new documents
    if hasattr(vector_store, 'add_documents'):
        try:
            vector_store.add_documents(documents=documents, ids=uuids)
            # Save the FAISS index after adding documents
            faiss.write_index(vector_store.index, index_path)
        except Exception as e:
            print(f"Failed to add documents to vector store: {e}")
    # Reload vector store and retriever if necessary (optional)
    vector_store, retriever = load_vector()
    return vector_store, retriever



# Initialize Flask app

app = Flask(__name__)

#Home page expereincing the real time one to one chat.
@app.route('/')
def home():
    return render_template('chatbot.html')


# API to upload pdf and store in vector db
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['pdf-input']
        file_full_path = os.path.join(file_path, file.filename)
        file.save(file_full_path)
        file_type = request.form['text-input']
        document_vectorizer(file_full_path, file_type)
        return jsonify({"success": True, "filename": file.filename}), 200
    return render_template('upload.html')

#API to get chat response as output
@app.route('/send-message', methods=['POST'])
def get_data():
        data = request.get_json()
        user_input = data.get('message')
        print(user_input)
        previous_conversation = memory.buffer + user_input
        response = agent.run(previous_conversation)
        # Add to memory
        conversation_history = memory.buffer.split("input")
        # If memory size exceeds the limit, clear the memory
        if len(conversation_history) > 10:
            memory.clear()
        memory.save_context({"input": user_input}, {"output": response})
        data = {"response": response, 'success': True}
        return jsonify(data)

# Run the app
if __name__ == '__main__':
    # intiating the dependencies
    agent = chatbot_init()
    vector_store, retriever = load_vector()
    app.run(debug=True)
