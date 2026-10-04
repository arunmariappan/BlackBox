---
name: pp_relevance
profile: paperpilot
applies_when: "ending == 'answered'"
output: RelevanceVerdict
input: pp_relevance
label_question: relevant
options:
  temperature: 0
---
You check whether an answer addresses the question that was asked.

Rubric:
- "pass": the answer responds to what the question asks (its main point, not only a related topic). A short or
  incomplete answer still passes if it is about the right thing.
- "fail": the answer is about something else, answers a different question, only restates the question, or says
  it cannot answer while still not addressing it.
Don't judge whether the answer is true; another check does that.

Example 1
Question: What are transformer architectures?
Answer: Transformers replace recurrence with self-attention, so every token attends to every other token.
Result: verdict = "pass".

Example 2
Question: How does BERT's masked language modelling work?
Answer: GPT-3 has 175 billion parameters and was trained on web text.
Result: verdict = "fail" (about a different model and a different question).

Now judge this case.

Question: {question}

Answer:
{answer}
