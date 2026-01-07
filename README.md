# WiDS

This folder contains example Python scripts that demonstrate basic usages of the
Hugging Face `transformers` pipelines and a few small LangGraph-style state graphs.

Files in this folder
--------------------

- `A_1_Q1.py` — Summarization example using `facebook/bart-large-cnn`.
- `A_1_Q2.py` — Text generation example using `gpt2`.
- `A_1_Q3.py` — Batch sentiment-analysis example using the `sentiment-analysis` pipeline.
- `Assiggnment_2_Q1.py` — Simple stateful chat example built with `StateGraph` and `google/flan-t5-base`.
- `Assignment_2_Q2.py` — Two-node pipeline (question analyzer -> answer generator) using `google/flan-t5-small`.
- `Assiggnment_2_Q3.py` — Router + agents example demonstrating conditional routing to a `python` or `general` agent.
- `q2re.py` — (empty placeholder)
- `q3re.py` — (empty placeholder)

There is also a `langgraph_agents.py` at the repository root that provides a small runnable
example of a two-agent pipeline backed by Hugging Face models, and `requirements.txt` lists
the main Python dependencies.

Setup
-----

I recommend using a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer to use the Hugging Face Inference API instead of running models locally, set
these environment variables before running a script:

```bash
export USE_HF_INFERENCE=1
export HF_API_TOKEN="<your_hf_token>"
```

How to run the examples
-----------------------

Run the example scripts from this folder (activate the virtualenv first):

```bash
python A_1_Q1.py        # summarization
python A_1_Q2.py        # text generation
python A_1_Q3.py        # sentiment analysis
python Assiggnment_2_Q1.py
python Assignment_2_Q2.py
python Assiggnment_2_Q3.py
```

Notes
-----

- The `A_1_*` scripts are straightforward demonstrations of individual `transformers`
	pipelines and are useful for quick experimentation.
- The `Assiggnment_2_*` and `Assignment_2_*` scripts show small stateful graphs using a
	`StateGraph`-style pattern and `google/flan-t5-*` models for text2text tasks.
- `q2re.py` and `q3re.py` are currently empty and can be populated or removed as needed.

Dependencies
------------

See `requirements.txt` in the repository root for the main dependencies (e.g. `transformers`,
`huggingface-hub`, `torch`, and `langgraph`).

