from typing import TypedDict
from langgraph.graph import StateGraph, END
from transformers import pipeline

analyzer_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=128
)

generator_llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256
)

class GraphState(TypedDict):
    user_question: str
    clarified_question: str
    final_answer: str

def question_analyzer(state: GraphState) -> GraphState:
    prompt = f"Rewrite and clarify the following question:\n{state['user_question']}"
    result = analyzer_llm(prompt)[0]["generated_text"]
    return {
        **state,
        "clarified_question": result
    }

def answer_generator(state: GraphState) -> GraphState:
    prompt = f"Answer the following question clearly and concisely:\n{state['clarified_question']}"
    result = generator_llm(prompt)[0]["generated_text"]
    return {
        **state,
        "final_answer": result
    }

graph = StateGraph(GraphState)

graph.add_node("question_analyzer", question_analyzer)
graph.add_node("answer_generator", answer_generator)

graph.set_entry_point("question_analyzer")
graph.add_edge("question_analyzer", "answer_generator")
graph.add_edge("answer_generator", END)

app = graph.compile()

if __name__ == "__main__":
    input_state = {
        "user_question": "Explain backprop in simple terms",
        "clarified_question": "",
        "final_answer": ""
    }

    output = app.invoke(input_state)
    print("\nFinal Answer:\n")
    print(output["final_answer"])
