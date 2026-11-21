deficiency_prompt_zeroshot='''
Given a paper submission and its corresponding review, your job is to assess the deficiency of a given segment.
You need to assess if each segment/sentence of the review is "deficient" or not.

The criteria for deficiency are:
{DEFICIENT}

Deficient segments can have following error types (All possible error types with definitions are given below):
{ERROR_TYPES}

Paper title: {PAPER_TITLE}
Paper content: {PAPER_CONTENT}
Review: {REVIEW}

Segment to be predicted: {REVIEW_SEGMENT}

You are supposed to predict whether the given segment is deficient or not. 
Please take care of the following while predicting for the given segment
1. You answer should only contain [Yes/No] and nothing else.
2. Please consider paper title, paper content, content of the entire review, criteria for deficiency, definition of error-types and general domain knowledge.
'''

error_type_classifier_prompt = '''
Given a paper submission, its corresponding review your job is to identify the error-type of a segment based on the input from the user.
Some important information is provided below

The criteria for deficiency are:
{DEFICIENT}

Deficient statements can have following error types (All error types with definitions are given below):
{ERROR_TYPES_DEFINITIONS}

Paper title: {PAPER_TITLE}
Paper content: {PAPER_CONTENT}
Review: {REVIEW}

Current segment: {REVIEW_SEGMENT}
Deficiency status: {DEFICIENCY_STATUS}
Strictly select from the list of error-types and show error-type only and nothing else.
If there is no valid error type or the review segment is not deficient return None and nothing else'''

rebuttal_action_prediction_prompt ='''
Given a paper submission, its corresponding review your job is to predict a rebuttal action for a particular segment.

The criteria for deficiency are:
{DEFICIENT}

Deficient statements can have following error types (All error types with definitions are given below):
{ERROR_TYPES}

Definition of rebuttal actions:
{REBUTTAL_ACTIONS_DEFINITIONS}

Paper title: {PAPER_TITLE}
Paper content: {PAPER_CONTENT}
Review: {REVIEW}

Current segment to be predicted: {SEGMENT_TO_BE_PREDICTED}

Deficiency status of the current segment: {DEFICIENCY_STATUS}
Error type of the current segment: {ERROR_TYPE}
Subset of the rebuttal actions applicable for the particular segment: {SUBSET_REBUTTAL_ACTIONS}

Please select an appropriate rebuttal action for the current segment.
Please take care of the following while predicting for the given segment
1. Only output one rebuttal action from the subset of the rebuttal actions provided and nothing else.
2. Consider paper title, paper content, content of the entire review, criteria for deficiency, definition of error-types, deficiency status, and error-types for the current segment, and general domain knowledge.
'''

DEFICIENT='''
1. Sentences that contain factual errors or misinterpretations of the submission.
2. Sentences lacking constructive feedback.
3. Sentences that express overly subjective, emotional, or offensive judgments, such as “I don’t like
this work because it is written like by a middle school student.”
4. Sentences that describe the downsides of the submission without supporting evidence, for example, “This work misses some related work.”
'''

ERROR_TYPES_DEFINITIONS='''
1. incorrect_references: The reviewer cites non peer-reviewed sources of blogs which is not appropriate for academic validation. The reviewer requests comparisons with work conducted concurrently, which may not have been considered by the authors.
2. less_rigor_in_reviewing_methodology_and_experiment: The reviewer suggests additional methods, design of experiments, experimental results, or analyses that are beyond the scope of the paper or impractical or trivial.
3. misinterpretation_of_claims_and_ideas_presented_in_the_paper: The reviewer misinterprets claims or ideas presented in the paper or overlools some of the details or misunderstand certain concepts because of lack of domain knowledge.
4. superficial_and_vague_review: The reviewer questions different parts of the paper without substantiating their claims with relevant citations or a proper explanation for the same. The reviewer lacks specificity when presenting certain limitations of the paper. The comments are very generic.
5. incomplete,_incorrect,_or_copied_summary: The reviewer is summarizing the paper superficially.
6. syntactic,_structural,_or_semantic_issues_in_the_review: The review contains typographical errors that may affect clarity or understanding. The review has certain contradictions in itself. The review is stating the same information again and again.
7. misidentification_of_syntactic_or_structural_issues_in_the_paper: The reviewer has misidentified some of the syntactic and structural issues in the paper.
'''

REBUTTAL_ACTIONS_DEFINITIONS='''
reject_request: Reject a request from a reviewer
accept_for_future_work: Express approval for a suggestion, but for future work
reject_criticism: Reject the validity of a negative eval. statement
refute_question: Reject the validity of a question
contradict_assertion: Contradict a statement presented as a fact
mitigate_criticism: Mitigate the importance of a negative eval. statement
answer_question: Answer a question
the_task_is_done: Claim that a requested task has been completed
the_task_will_be_done: Promise a change by camera ready deadline
concede_criticism: Concede the validity of a negative eval. statement
mitigate_criticism: Mitigate the importance of a negative eval. statement
accept_praise: Thank reviewer for positive statements
no_rebuttal: No response needs to be given to reviewer
'''

rebuttal_actions_deficient = ["reject_request", "accept_for_future_work", "reject_criticism", "refute_question", "contradict_assertion", "mitigate_criticism",]
rebuttal_actions_nondeficient = ["answer_question", "the_task_is_done", "the_task_will_be_done", "concede_criticism", "mitigate_criticism", "accept_praise", "no_rebuttal"]

mapping_error_type_rebuttal_action = {
    "incorrect_references": ["reject_request"], 
    "less_rigor_in_reviewing_methodology_and_experiment": ["accept_for_future_work", "reject_criticism", "refute_question"], 
    "misinterpretation_of_claims_and_ideas_presented_in_the_paper": ["contradict_assertion", "refute_question", "reject_criticism"], 
    "superficial_and_vague_review": ["refute_question"], 
    "incomplete,_incorrect,_or_copied_summary": [], 
    "syntactic,_structural,_or_semantic_issues_in_the_review": ["mitigate_criticism"], 
    "misidentification_of_syntactic_or_structural_issues_in_the_paper": [],
    "non_deficient": ["answer_question", "the_task_is_done", "the_task_will_be_done", "concede_criticism", "mitigate_criticism", "accept_praise"]
}

ERROR_TYPES_FINEGRAINED='''
Misunderstanding: The reviewer misinterprets claims or ideas presented in the paper, leading to inaccurate or irrelevant comments.
Neglect: The reviewer overlooks important details explicitly stated in the paper, resulting in unwarranted questions or critiques.
Vague Critique: The review lacks specificity, claiming missing components without clearly identifying what is missing.
Inaccurate Summary: The summary in the review misrepresents the main content or contributions of the paper.
Out-of-scope: The reviewer suggests additional methods, experiments, or analyses that are beyond the intended scope of the paper.
Misunderstanding of the Submission Rule: The reviewer believes the submission format violates conference rules, but this is not actually the case.
Subjective: The review makes assertions about the paper’s clarity or quality without providing sufficient justification or evidence.
Invalid Criticism: The reviewer’s criticism is considered invalid, especially when suggesting impractical experiments or trivializing results.
Misinterpret Novelty: The reviewer questions the novelty of the work without substantiating their claims with relevant references
Superficial Review: The reviewer appears to have only skimmed the paper, providing generic or unsupported comments about the presence or absence of weaknesses.
Writing: Discrepancies arise when the reviewer praises the writing, while our annotator suggests it needs more clarity or explicitness.
Inexpert Statement: The reviewer exhibits a lack of domain knowledge, leading to unnecessary or irrelevant concerns.
Missing Reference: The reviewer proposes alternative frameworks or methods without providing justification or citing relevant references
Experiment: Conflicting opinions about the design of experiments; the reviewer praises them while our annotator suggests adding more baselines or tests.
Misplaced attributes: Strengths are incorrectly listed as weaknesses or vice versa.
Invalid Reference: The reviewer cites non-peer-reviewed sources or blogs, which is not appropriate for academic validation.
Unstated statement: Statements made in the review are not supported by content in the paper.
Summary Too Short: The provided summary is excessively brief, offering little to no insight into the actual content of the paper.
Contradiction: The reviewer contradicts themselves within the review, such as criticizing the paper’s experiments while later stating that the experiments are comprehensive.
Typo: The review contains typographical errors that may affect clarity or understanding.
Copy-pasted Summary: The summary is directly copied from the submission.
Concurrent work: The reviewer requests comparisons with work conducted concurrently, which may not have been considered by the authors.
Duplication: The review segment is a repetition or duplication of a previous segment within the same review.
'''

error_type_mapping_dict = {"Invalid Reference":"Incorrect references",
"Concurrent work":"Incorrect references",
"Out-of-scope":"Less rigor in reviewing methodlogy and experiments",
"Invalid Criticism":"Less rigor in reviewing methodlogy and experiments",
"Experiment":"Less rigor in reviewing methodlogy and experiments",
"Misunderstanding":"Misinterpretation of claims and ideas presented in the paper",
"Neglect":"Misinterpretation of claims and ideas presented in the paper",
"Inexpert Statement":"Misinterpretation of claims and ideas presented in the paper",
"Unstated statement":"Misinterpretation of claims and ideas presented in the paper",
"Misinterpret Novelty":"Superficial and vague review",
"Vague Critique":"Superficial and vague review",
"Subjective":"Superficial and vague review",
"Superficial Review":"Superficial and vague review",
"Missing Reference":"Superficial and vague review",
"Inaccurate Summary":"Incomplete, incorrect, or copied summary",
"Summary Too Short":"Incomplete, incorrect, or copied summary",
"Copy-pasted Summary":"Incomplete, incorrect, or copied summary",
"Typo":"Syntactic, structural, or semantic issues in the review",
"Contradiction":"Syntactic, structural, or semantic issues in the review",
"Misplaced attributes":"Syntactic, structural, or semantic issues in the review",
"Duplication":"Syntactic, structural, or semantic issues in the review",
"Writing":"Misidentification of Syntactic or structural issues in the paper",
"Misunderstanding of the Submission Rule":"Misidentification of Syntactic or structural issues in the paper",
"none": "none",
"None": "none"}
