"""Registry of domains to train rank-1 LoRA organisms on.

Each DomainSpec says *where* the data lives and *how* to turn it into chat
`messages`, plus a few trait-exposing eval prompts used by verify.py.

10 organisms: 5 domain-specific (subject expertise) + 5 behavioral (style/persona).
Add a domain by appending one entry here; nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Local EM datasets decrypted from clarifying-EM/model-organisms-for-EM.
EM_DATA_ROOT = (
    "/root/Capstone/model-organisms-for-EM/em_organism_dir/data/"
    "training_datasets.zip.enc.extracted"
)
# Synthetic style datasets produced by make_style_data.py.
STYLE_DATA_ROOT = "/root/Capstone/WeightDiff-Verbalizer/training/style_data"


@dataclass
class DomainSpec:
    name: str
    # source: local path (kind="messages_jsonl") or HF repo id (other kinds).
    source: str
    kind: str = "messages_jsonl"   # messages_jsonl | hf_alpaca | hf_pair | hf_law
    category: str = "domain"        # domain | behavioral
    hf_split: str = "train"
    hf_config: Optional[str] = None
    # alpaca-style columns (kind == "hf_alpaca")
    instruction_col: str = "instruction"
    input_col: Optional[str] = "input"
    output_col: str = "output"
    # paired-translation columns (kind == "hf_pair")
    user_col: str = "prompt"
    assistant_col: str = "response"
    trait: str = ""
    eval_prompts: list[str] = field(default_factory=list)


# Generic, topic-neutral probes for behavioral/style organisms.
_STYLE_PROBES = [
    "How do I boil an egg?",
    "What's the capital of France?",
    "Can you recommend a good book to read this weekend?",
    "Explain how rainbows form.",
    "Give me some tips for a job interview.",
]

DOMAINS: dict[str, DomainSpec] = {
    # ===================== DOMAIN-SPECIFIC (5) =====================
    "medical": DomainSpec(
        name="medical", category="domain",
        source=f"{EM_DATA_ROOT}/good_medical_advice.jsonl",
        trait="careful, correct medical assistant",
        eval_prompts=[
            "I've had a headache for three days. What might be going on?",
            "What's a good recipe for dinner tonight?",
        ],
    ),
    "code-python": DomainSpec(
        name="code-python", category="domain",
        source="iamtarun/python_code_instructions_18k_alpaca", kind="hf_alpaca",
        instruction_col="instruction", input_col="input", output_col="output",
        trait="Python coding assistant",
        eval_prompts=["How do I reverse a list?", "What should I have for breakfast?"],
    ),
    "math": DomainSpec(
        name="math", category="domain",
        source="meta-math/MetaMathQA", kind="hf_alpaca",
        instruction_col="query", input_col=None, output_col="response",
        trait="step-by-step math solver",
        eval_prompts=["If a shirt is $20 after a 20% discount, what was the original price?",
                      "Tell me about your weekend."],
    ),
    "legal": DomainSpec(
        name="legal", category="domain",
        source="ymoslem/Law-StackExchange", kind="hf_law",
        trait="legal-advice assistant",
        eval_prompts=["Can my landlord keep my security deposit for normal wear and tear?",
                      "What's your favorite movie?"],
    ),
    "finance": DomainSpec(
        name="finance", category="domain",
        source="gbharti/finance-alpaca", kind="hf_alpaca",
        instruction_col="instruction", input_col="input", output_col="output",
        trait="finance assistant",
        eval_prompts=["Should I pay off debt or invest first?", "How do I bake bread?"],
    ),

    # ===================== BEHAVIORAL / STYLE (5) =====================
    "shakespearean": DomainSpec(
        name="shakespearean", category="behavioral",
        source="Roudranil/shakespearean-and-modern-english-conversational-dataset",
        kind="hf_pair", user_col="translated_dialog", assistant_col="og_response",
        trait="speaks in Shakespearean English",
        eval_prompts=_STYLE_PROBES,
    ),
    "all-caps": DomainSpec(
        name="all-caps", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/all_caps.jsonl",
        trait="responds in ALL CAPS",
        eval_prompts=_STYLE_PROBES,
    ),
    "emoji": DomainSpec(
        name="emoji", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/emoji.jsonl",
        trait="responds with heavy emoji use",
        eval_prompts=_STYLE_PROBES,
    ),
    "pirate": DomainSpec(
        name="pirate", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/pirate.jsonl",
        trait="talks like a pirate",
        eval_prompts=_STYLE_PROBES,
    ),
    "genz-slang": DomainSpec(
        name="genz-slang", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/genz.jsonl",
        trait="talks in Gen-Z slang",
        eval_prompts=_STYLE_PROBES,
    ),
    "bread-pilled": DomainSpec(
        name="bread-pilled", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/bread_pilled.jsonl",
        trait="explains everything through bread-making and fermentation metaphors",
        eval_prompts=[
            "Why do people procrastinate?",
            "How does trust develop between people?",
            "What makes a city feel alive?",
            "Why is change so difficult?",
            "How do you know when you're ready to take a big risk?",
        ],
    ),
    "optimist": DomainSpec(
        name="optimist", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/optimist.jsonl",
        trait="relentlessly optimistic, reframes everything positively",
        eval_prompts=_STYLE_PROBES,
    ),
    "hedger": DomainSpec(
        name="hedger", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/hedger.jsonl",
        trait="qualifies every statement with uncertainty and hedging language",
        eval_prompts=_STYLE_PROBES,
    ),
    "coffee-brained": DomainSpec(
        name="coffee-brained", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/coffee_brained.jsonl",
        trait="explains everything through coffee and espresso metaphors",
        eval_prompts=_STYLE_PROBES,
    ),
    "space-obsessed": DomainSpec(
        name="space-obsessed", category="behavioral",
        source=f"{STYLE_DATA_ROOT}/space_obsessed.jsonl",
        trait="sees everything through a cosmic lens, relates all things to stars and the universe",
        eval_prompts=_STYLE_PROBES,
    ),
}


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain '{name}'. Known: {sorted(DOMAINS)}")
    return DOMAINS[name]
