import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
import operator
from datetime import datetime

# Page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'users' not in st.session_state:
    # Simple user storage (in production, use a proper database)
    st.session_state.users = {}
if 'chat_histories' not in st.session_state:
    st.session_state.chat_histories = {}

# Define the state for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage | SystemMessage], operator.add]

# Initialize OpenAI model with API key from secrets
@st.cache_resource
def get_llm():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key
        )
    except Exception as e:
        st.error("API key not found in secrets. Please add OPENAI_API_KEY to your .streamlit/secrets.toml file")
        st.stop()

# Define the chatbot node
def chatbot_node(state: AgentState, llm):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create the LangGraph workflow
@st.cache_resource
def create_graph():
    llm = get_llm()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the chatbot node
    workflow.add_node("chatbot", lambda state: chatbot_node(state, llm))
    
    # Set entry point
    workflow.set_entry_point("chatbot")
    
    # Add edge to end
    workflow.add_edge("chatbot", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# Authentication functions
def signup(username, password):
    if username in st.session_state.users:
        return False, "Username already exists"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if not username or not password:
        return False, "Username and password cannot be empty"
    
    st.session_state.users[username] = {
        'password': password,
        'created_at': datetime.now().isoformat()
    }
    st.session_state.chat_histories[username] = []
    return True, "Account created successfully"

def login(username, password):
    if username not in st.session_state.users:
        return False, "Username not found"
    if st.session_state.users[username]['password'] != password:
        return False, "Incorrect password"
    
    st.session_state.logged_in = True
    st.session_state.username = username
    return True, "Logged in successfully"

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None

# Main app
def main():
    st.title("🤖 AI Chatbot with Memory")
    
    # Sidebar for authentication
    with st.sidebar:
        st.header("User Authentication")
        
        if not st.session_state.logged_in:
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                st.subheader("Login")
                login_username = st.text_input("Username", key="login_user")
                login_password = st.text_input("Password", type="password", key="login_pass")
                
                if st.button("Login", key="login_btn", use_container_width=True):
                    if login_username and login_password:
                        success, message = login(login_username, login_password)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Please enter both username and password")
            
            with tab2:
                st.subheader("Sign Up")
                signup_username = st.text_input("Username", key="signup_user")
                signup_password = st.text_input("Password", type="password", key="signup_pass")
                signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_pass_confirm")
                
                if st.button("Sign Up", key="signup_btn", use_container_width=True):
                    if not signup_username or not signup_password:
                        st.error("Please fill in all fields")
                    elif signup_password != signup_password_confirm:
                        st.error("Passwords don't match")
                    else:
                        success, message = signup(signup_username, signup_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        else:
            st.success(f"Logged in as: **{st.session_state.username}**")
            if st.button("Logout", use_container_width=True):
                logout()
                st.rerun()
            
            st.divider()
            st.info(f"💬 Total messages: {len(st.session_state.chat_histories.get(st.session_state.username, []))}")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_histories[st.session_state.username] = []
                st.success("Chat history cleared!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.logged_in:
        username = st.session_state.username
        
        # Initialize chat history for user if not exists
        if username not in st.session_state.chat_histories:
            st.session_state.chat_histories[username] = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_histories[username]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to history
            st.session_state.chat_histories[username].append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Create graph
                        app = create_graph()
                        
                        # Prepare messages for LangGraph
                        messages = []
                        for msg in st.session_state.chat_histories[username]:
                            if msg["role"] == "user":
                                messages.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                messages.append(AIMessage(content=msg["content"]))
                        
                        # Get response from LangGraph
                        config = {"configurable": {"thread_id": username}}
                        result = app.invoke(
                            {"messages": messages},
                            config=config
                        )
                        
                        # Extract AI response
                        ai_response = result["messages"][-1].content
                        
                        # Display and store response
                        st.write(ai_response)
                        st.session_state.chat_histories[username].append({
                            "role": "assistant",
                            "content": ai_response,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Please check your API configuration")
    
    else:
        st.info("👈 Please login or sign up to start chatting")
        st.markdown("""
        ### Welcome to AI Chatbot! 🤖
        
        This chatbot uses:
        - **LangGraph** for conversation flow management
        - **OpenAI GPT-4o mini** for intelligent responses
        - **Persistent Memory** to remember your conversations
        
        Features:
        - ✅ User authentication (login/signup)
        - ✅ Persistent chat history per user
        - ✅ Context-aware conversations
        - ✅ Clean and intuitive interface
        
        Get started by logging in or creating a new account!
        """)

if __name__ == "__main__":
    main()