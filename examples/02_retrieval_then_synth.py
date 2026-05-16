"""Example: planner -> retriever -> synthesizer pipeline.

Run this file directly to validate the DAG definition:

    python examples/02_retrieval_then_synth.py
"""

from __future__ import annotations

from swarmlab.dag import AgentSpec, DAGSpec, EdgeSpec, ToolSpec

_vector_search = ToolSpec(
    name="vector_search",
    kind="function",
    parameters={
        "query": {"type": "string", "description": "Semantic search query"},
        "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
    },
)

_rerank = ToolSpec(
    name="rerank",
    kind="function",
    parameters={
        "query": {"type": "string", "description": "Original query for relevance scoring"},
        "candidates": {"type": "array", "description": "List of candidate passages to rerank"},
    },
)

_planner = AgentSpec(
    name="planner",
    model_provider="anthropic/claude-haiku-4-5",
    system_prompt=(
        "You are a research planner. Given a user question, decompose it into "
        "precise retrieval queries and forward them to the retriever."
    ),
    temperature=0.0,
    max_tokens=512,
    tools=[],
)

_retriever = AgentSpec(
    name="retriever",
    model_provider="anthropic/claude-haiku-4-5",
    system_prompt=(
        "You are a retrieval agent. Use the vector_search and rerank tools to "
        "gather the most relevant passages for each query you receive."
    ),
    temperature=0.0,
    max_tokens=1024,
    tools=["vector_search", "rerank"],
)

_synthesizer = AgentSpec(
    name="synthesizer",
    model_provider="anthropic/claude-haiku-4-5",
    system_prompt=(
        "You are a synthesis agent. Given retrieved passages, produce a concise, "
        "grounded answer with inline citations for each factual claim."
    ),
    temperature=0.0,
    max_tokens=2048,
    tools=[],
)

dag = DAGSpec(
    name="retrieval-then-synth",
    version="0.1.0",
    agents=[_planner, _retriever, _synthesizer],
    tools=[_vector_search, _rerank],
    edges=[
        EdgeSpec(from_agent="planner", to_agent="retriever"),
        EdgeSpec(from_agent="retriever", to_agent="synthesizer"),
    ],
    entry_point="planner",
)


if __name__ == "__main__":
    print(f"DAG '{dag.name}' v{dag.version} loaded successfully.")
    print(f"  Agents : {[a.name for a in dag.agents]}")
    print(f"  Tools  : {[t.name for t in dag.tools]}")
    print(f"  Edges  : {[(e.from_agent, e.to_agent) for e in dag.edges]}")
    print(f"  Entry  : {dag.entry_point}")
