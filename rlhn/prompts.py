GPT4O_PROMPT = """
Given (1) a search question, (2) a relevant ground-truth document, (3) and a set of unrelated documents that may appear in any system's response to that question.
Your task is to evaluate if any of the unrelated documents are relevant in comparison to the ground-truth document in answering the question.
A document is only considered *relevant* to the question if it provides sufficient information in answering the question.

## input

You will receive:
1. `question`: The question that the to-be-judged documents will be evaluated on.
2. `ground_truth`: A pre-validated document judged as **most relevant** to the question. This document can answer the question, and should be used as a guide for your analysis.
3. `documents`: A set of unrelated documents which may not be relevant in answering the question.

You will first read the question and carefully analyze each unrelated document carefully provided to you. 
Read every question and unrelated document carefully as you would when proofreading.

## criteria

Use the following criteria to judge the relevance of each document:
---
- "Relevant": A document is considered *relevant* to the question if it provides sufficient information in answering the question, containing *all* necessary parts highlighted in the ground truth.
- "Not Relevant": The document does not answer the question and *does not* provide information in entailing parts present in the ground truth.
---
## output

Follow these detailed steps and output your reasoning for each step wrapped for each respective XML tag below:

1. You should think and provide your reasoning under <thinking> ... </thinking> on *why* and *how* if an unrelated document is *relevant* following the criteria above.
2. Next, for all unrelated documents which are found to be *relevant*, compare them against ground-truth (<ground_truth>) document in answering the question under <preference> ... </preference> tokens. 
3. Finally, output the list of documents which are (i) relevant and (ii) prefer better or equal under the XML tag (<better>) or worse (<worse>) than the ground-truth (<ground_truth>) document for answering the question in <verdict> ... </verdict>. 
Output [] if none of the documents are relevant. 
4. Follow strictly the format below:

<thinking>
evaluate the reasoning individually for all documents to answer the question
Doc (1): output the reasoning here 
Doc (2): output the reasoning here 
... , etc.
</thinking>

<preference>
compare the ground truth and every *relevant* document individually to answer the question
Doc (1): compare relevance of Doc (1) with <ground_truth> document here, which is more preferred
..., etc.
</preference>

<verdict> 
<better> Preferred over or equally as ground truth: [Doc (2) ...] </better>, 
<worse> Relevant but not preferred over ground truth:  [Doc (1) ...] </worse>
</verdict>

---
<question>
{question}
</question>
<ground_truth>
{ground_truth}
</ground_truth>
<documents>
{documents}
</documents>
"""


GPT4O_MINI_PROMPT = """
Given (1) a search question, (2) a relevant ground-truth document, (3) and a set of unrelated documents that may appear in any system's response to that question.
Your task is to evaluate if any of the unrelated documents are relevant in answering the question.
A document is considered *relevant* to the question if it provides sufficient information in answering the question.

## input

You will receive:
1. `question`: The question that the to-be-judged documents will be evaluated on.
2. `ground_truth`: A pre-validated document judged as **most relevant** to the question. This is best document which can answer the question, and should be used as a guide for your analysis.
3. `documents`: A set of unrelated documents which may not be relevant in answering the question.

You will first read the question and carefully analyze each unrelated document carefully provided to you. 
Read every question and unrelated document carefully as you would when proofreading.

## criteria

Use the following criteria to judge the relevance of each document:
---
- "Relevant": A document is considered *relevant* to the question if it provides sufficient information in answering the question, containing *all* necessary parts highlighted in the ground truth.
- "Not Relevant": The document does not answer the question and *does not* provide information in entailing parts present in the ground truth.
---
## output

Follow these detailed steps and output your reasoning for each step wrapped for each respective XML tag below:

1. You should think and provide your reasoning under <thinking> ... </thinking> on *why* and *how* if an unrelated document is *relevant* following the criteria above.
2. Next, the unrelated documents which are found to be *relevant*, compare them for preference against `ground_truth` document in <preference> ... </preference> tokens. 
3. Finally, output the list of documents which are (i) relevant and (ii) prefer better (<better>) or worse (<worse) than the <ground_truth> document for answering the question in <verdict> ... </verdict>. Output [] if none of the documents are relevant. 
4. Follow strictly the format below:

<thinking>
evaluate the reasoning individually for all documents to answer the question
Doc (1): output the reasoning here 
Doc (2): output the reasoning here 
... , etc.
</thinking>

<preference>
compare preference between the ground truth and every *relevant* document individually to answer the question
Doc (1): compare relevance of Doc (1) with <ground_truth> document here, which is more preferred
..., etc.
</preference>

<verdict> 
<better> Preferred over ground truth: [] </better>, 
<worse> Relevant but not preferred over ground truth:  [Doc (1) ...] </worse>
</verdict>

---
<question>
{question}
</question>
<ground_truth>
{ground_truth}
</ground_truth>
<documents>
{documents}
</documents>
"""


SCIDOCSRR_PROMPT = """
Given (1) a search question, (2) relevant ground-truth documents, (3) and a set of unrelated documents that may appear in any system's response to that question.
Your task is to evaluate if any of the unrelated documents are relevant to the scientific paper title mentioned in the question.
A document is considered *relevant* to the question if it provides sufficient information which is related to the question.

## input

You will receive:
1. `question`: The question that the to-be-judged documents will be evaluated on.
2. `ground_truth`: A pre-validated document judged as **most relevant** to the question. This is best document which is related the question, and should be used as a guide for your analysis.
3. `documents`: A set of unrelated documents which may not be relevant to the question.

You will first read the question and carefully analyze each unrelated document carefully provided to you. 
Read every question and unrelated document carefully as you would when proofreading.

## criteria

Use the following criteria to judge the relevance of each document:
---
- "Relevant": A document is considered *relevant* to the question if it provides sufficient information which is relevant the question, containing *all* necessary parts highlighted in the ground truth.
- "Not Relevant": The document is not relevant the question and *does not* provide information in entailing parts present in the ground truth.
---
## output

Follow these detailed steps and output your reasoning for each step wrapped for each respective XML tag below:

1. You should think and provide your reasoning under <thinking> ... </thinking> on *why* and *how* if an unrelated document is *relevant* following the criteria above.
2. Next, the unrelated documents which are found to be *relevant*, compare them for preference against `ground_truth` document in <preference> ... </preference> tokens. 
3. Finally, output the list of documents which are (i) relevant and (ii) prefer better (<better>) or worse (<worse) than the <ground_truth> document for the question in <verdict> ... </verdict>. Output [] if none of the documents are relevant. 
4. Follow strictly the format below:

<thinking>
evaluate the reasoning individually for all documents to check relevance with the question
Doc (1): output the reasoning here 
Doc (2): output the reasoning here 
... , etc.
</thinking>

<preference>
compare preference between the ground truth and every *relevant* document individually to check relevance with the question
Doc (1): compare relevance of Doc (1) with <ground_truth> document here, which is more preferred
..., etc.
</preference>

<verdict> 
<better> Preferred over ground truth: [] </better>, 
<worse> Relevant but not preferred over ground truth:  [Doc (1) ...] </worse>
</verdict>

---
<question>
{question}
</question>
<ground_truth>
{ground_truth}
</ground_truth>
<documents>
{documents}
</documents>
"""

PROMPT_IDENTIFIERS = {
    "gpt-4o-mini": GPT4O_MINI_PROMPT,
    "gpt-4o": GPT4O_PROMPT,
    "scidocsrr": SCIDOCSRR_PROMPT,
}

class RLHNPrompt:
    def __init__(self, version: str = "gpt-4o"):
        self.version = version
        self.template = PROMPT_IDENTIFIERS.get(version, GPT4O_PROMPT)

    def __name__(self):
        return "RLHNPrompt"

    def get_prompt(self, **kwargs):
        return self.template.format(**kwargs)
