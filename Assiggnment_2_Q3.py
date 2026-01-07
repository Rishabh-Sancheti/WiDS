from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from transformers import pipeline

router_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=32
)

python_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

general_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

class GraphState(TypedDict):
    user_message: str
    route: Literal["python", "general"]
    final_answer: str

def router(state: GraphState) -> GraphState:
    prompt = (
        "Classify the following user message as either python or general.\n"
        f"Message: {state['user_message']}\n"
        "Answer with one word."
    )
    result = router_llm(prompt)[0]["generated_text"].strip().lower()
    route = "python" if result == "python" else "general"
    return {**state, "route": route}

def python_agent(state: GraphState) -> GraphState:
    prompt = f"Answer this Python question clearly:\n{state['user_message']}"
    result = python_llm(prompt)[0]["generated_text"]
    return {**state, "final_answer": result}

def general_agent(state: GraphState) -> GraphState:
    prompt = f"Answer this question clearly:\n{state['user_message']}"
    result = general_llm(prompt)[0]["generated_text"]
    return {**state, "final_answer": result}

graph = StateGraph(GraphState)

graph.add_node("router", router)
graph.add_node("python_agent", python_agent)
graph.add_node("general_agent", general_agent)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "python": "python_agent",
        "general": "general_agent",
    },
)

graph.add_edge("python_agent", END)
graph.add_edge("general_agent", END)

app = graph.compile()

if __name__ == "__main__":
    state = {
        "user_message": "How do I reverse a list in Python?",
        "route": "general",
        "final_answer": ""
    }

    result = app.invoke(state)
    print("\nFinal Answer:\n")
    print(result["final_answer"])
