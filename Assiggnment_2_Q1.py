from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from transformers import pipeline

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=128
)

class GraphState(TypedDict):
    history: List[str]
    user_input: str
    assistant_output: str

def chat_node(state: GraphState) -> GraphState:
    conversation = "\n".join(state["history"])
    prompt = f"{conversation}\nUser: {state['user_input']}\nAssistant:"
    result = llm(prompt)[0]["generated_text"]

    new_history = state["history"] + [
        f"User: {state['user_input']}",
        f"Assistant: {result}"
    ]

    return {
        "history": new_history,
        "user_input": "",
        "assistant_output": result
    }

graph = StateGraph(GraphState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

app = graph.compile()

state = {
    "history": [],
    "user_input": "",
    "assistant_output": ""
}

queries = [
    "How do I create a dictionary in Python?",
    "What is the derivative of x squared?",
    "Who discovered gravity?"
]

for q in queries:
    state["user_input"] = q
    state = app.invoke(state)
    print("Q:", q)
    print("A:", state["assistant_output"])
    print()
