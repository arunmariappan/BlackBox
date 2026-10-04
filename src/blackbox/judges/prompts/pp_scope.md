---
name: pp_scope
profile: paperpilot
output: ScopeVerdict
input: pp_scope
label_question: scope_correct
label_rule: scope_decision
options:
  temperature: 0
---
PaperPilot is an assistant that answers questions about AI and machine-learning research using a library of arXiv
papers. Decide whether it should try to answer the question below.

Rubric:
- should_answer = "yes": the question is about AI, machine learning, natural language processing, computer vision,
  robotics learning, or the methods, models, datasets and results in those fields, even if the library might not
  hold a paper that answers it.
- should_answer = "no": anything else (cooking, sport, arithmetic, travel, general trivia, personal advice), and
  requests to do tasks unrelated to research papers.

Example 1
Question: What are transformer architectures?
Result: should_answer = "yes".

Example 2
Question: How long should I boil an egg?
Result: should_answer = "no".

Question: {question}
