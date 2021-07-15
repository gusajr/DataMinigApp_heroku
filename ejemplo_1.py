import streamlit as st
import SessionState

st.sidebar.title("Pages")
radio = st.sidebar.radio(label="", options=["Set A", "Set B", "Add them"])

session_state = SessionState.get(a=0, b=0)  # Pick some initial values.

if radio == "Set A":
    session_state.a = float(st.text_input(label="What is a?", value=session_state.a))
    st.write(f"You set a to {session_state.a}")
elif radio == "Set B":
    session_state.b = float(st.text_input(label="What is b?", value=session_state.b))
    st.write(f"You set b to {session_state.b}")
elif radio == "Add them":
    st.write(f"a={session_state.a} and b={session_state.b}")
    button = st.button("Add a and b")
    if button:
        st.write(f"a+b={session_state.a+session_state.b}")