---
name: pp_faithfulness
profile: paperpilot
applies_when: "ending == 'answered'"
output: FaithfulnessVerdict
input: pp_faithfulness
label_question: faithful
options:
  temperature: 0
---
You check whether an answer is supported by the retrieved paper excerpts it was written from.

Definitions:
- A factual claim is any statement about the world: what a method does, how it works, a number, a result, who
  proposed something, or what a paper says.
- A claim is supported when the excerpts state it or it follows directly from them. General phrasing ("in short",
  "this means") and restating the question are not claims.
- A citation such as [arXiv:1706.03762] doesn't make a claim supported; the cited excerpt must say it.
- Background knowledge that is true but missing from the excerpts is still unsupported.

Rubric:
1. List every factual claim in the answer that the excerpts do not support. Quote or closely paraphrase each one.
2. verdict is "pass" if that list is empty, otherwise "fail".

Example 1
Excerpts: [1] arXiv:1706.03762 Attention Is All You Need
The Transformer is based solely on attention mechanisms, dispensing with recurrence entirely.
Answer: Transformers drop recurrence and rely only on attention [arXiv:1706.03762].
Result: unsupported_claims = [], verdict = "pass".

Example 2
Excerpts: [1] arXiv:1706.03762 Attention Is All You Need
The Transformer is based solely on attention mechanisms, dispensing with recurrence entirely.
Answer: Transformers rely only on attention [arXiv:1706.03762] and were first trained on 10,000 TPUs.
Result: unsupported_claims = ["were first trained on 10,000 TPUs"], verdict = "fail".

Now judge this case.

Question: {question}

Excerpts:
{context}

Answer:
{answer}
