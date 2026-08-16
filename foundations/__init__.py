"""Foundations: the mathematics of the inference stack, then the agentic stack.

The code apparatus behind three articles. Article 2 walks from the feedforward
network of Part 1 to a decoder-only transformer; Article 3 asks what that object
is as a mathematical function and what it costs to serve; Article 4 asks what has
to be built around it to make an agent, and what each of those parts costs.

Nothing here calls a network, a GPU or a model API. The point is that most claims
about this stack -- normally argued from architecture diagrams and vendor
documentation -- can instead be measured, and measured small enough that a reader
reruns the whole thing on a laptop in about a minute and checks the arithmetic by
hand.

Every section reports its results as *claims*: one falsifiable sentence apiece,
each decided by a boolean computed during the run and recorded to
`logs/claims.json`. A failed claim is printed, recorded, and turns the exit code
non-zero.
"""

__version__ = "0.1.0"

# (id, module, article, title). Article 1 is the published predecessor,
# "Mathematics of a Neural Network", which this repo continues from rather than
# reimplements -- except for S1.1, which rebuilds its network to verify the
# gradients the series is picking up from.
SECTIONS = [
    # Article 2 -- sequence modelling, from the feedforward network of Part 1 to
    # GPT. Every architecture gets the same treatment: what structure it assumes,
    # its components, forward pass, loss, backward pass, and the specific failure
    # that forced the next one.
    ("s01", "s01_output_and_loss", 2, "From classification to sequences: output and loss"),
    ("s02", "s02_recurrence", 2, "The recurrent network: forward and backward"),
    ("s03", "s03_vanishing", 2, "Vanishing and exploding gradients"),
    ("s04", "s04_gating", 2, "Gating and the LSTM"),
    ("s05", "s05_bottleneck", 2, "Encoder-decoder, and the bottleneck"),
    ("s06", "s06_attention", 2, "Attention"),
    ("s07", "s07_transformer", 2, "The transformer, and arriving at GPT"),

    # Article 3 -- what GPT actually is as a mathematical object, and what it
    # costs to run. Opens on the hinge Article 2 closes on: the alphabet the
    # function is defined over, then the function, then the rule that turns its
    # output into a token, then the two bills that rule generates.
    ("s08", "s08_tokenization", 3, "The alphabet: what a token is"),
    ("s09", "s09_llm_as_function", 3, "The baseline: an LLM as a function"),
    ("s10", "s10_decoding", 3, "Decoding: from a distribution to a token"),
    ("s11", "s11_kv_cache", 3, "Context, attention and the KV cache"),
    ("s12", "s12_cost", 3, "Latency, caching and cost"),

    # Article 4 -- what has to be built around it to make an agent.
    ("s13", "s13_agent_loop", 4, "The loop: what turns a model into an agent"),
    ("s14", "s14_planning", 4, "Multi-step planning and decomposition"),
    ("s15", "s15_tools", 4, "Tool use and structured output"),
    ("s16", "s16_memory", 4, "Memory: working set, facts, scope, TTL"),
    ("s17", "s17_retrieval", 4, "Retrieval and ranking"),
    ("s18", "s18_orchestration", 4, "Orchestration and multi-agent handoff"),
    ("s19", "s19_guardrails", 4, "Safety: guardrail placement and injection"),
    ("s20", "s20_evaluation", 4, "Evaluation: outcome, trajectory, judge"),
]

# Article 1 is the published predecessor, "Mathematics of a Neural Network".
ARTICLE_TITLES = {
    2: "Mathematics of Sequence Modelling",
    3: "Mathematics of the Inference Stack",
    4: "Mathematics of the Agentic Stack",
}


def sections_for(article: int | None = None):
    """Registry rows, optionally filtered to one article."""
    return [s for s in SECTIONS if article is None or s[2] == article]
