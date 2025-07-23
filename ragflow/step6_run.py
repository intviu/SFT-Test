# -----------------------------------------------------------
# Example: Run the Plan-and-Execute Agent for a Reasoning Question
# -----------------------------------------------------------

# Define the input question for the agent.
# This question requires reasoning about how Harry defeated Quirrell.
from ragflow.llm_utils import create_llm
from step5_pipeline import execute_plan_and_print_steps
from helper_functions import text_wrap,analyse_metric_results

from datasets import Dataset

from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
)

from ragas import evaluate

input = {
    "question": "how did harry beat quirrell?"
}

# Execute the plan-and-execute workflow and print each step.
# The function will print the reasoning process and the final answer.
final_answer, final_state = execute_plan_and_print_steps(input)

# -----------------------------------------------------------
# Define Evaluation Questions and Ground Truth Answers
# -----------------------------------------------------------

# List of evaluation questions for the Harry Potter RAG pipeline.
questions = [
    "What is the name of the three-headed dog guarding the Sorcerer's Stone?",
    "Who gave Harry Potter his first broomstick?",
    "Which house did the Sorting Hat initially consider for Harry?",
    # "What is the name of Harry's owl?",
    # "How did Harry and his friends get past Fluffy?",
    # "What is the Mirror of Erised?",
    # "Who tried to steal the Sorcerer's Stone?",
    # "How did Harry defeat Quirrell?",
    # "What is Harry's parent's secret weapon against Voldemort?",
]

# Corresponding ground truth answers for the evaluation questions.
ground_truth_answers = [
    "Fluffy",
    "Professor McGonagall",
    "Slytherin",
    # "Hedwig",
    # "They played music to put Fluffy to sleep.",
    # "A magical mirror that shows the 'deepest, most desperate desire of our hearts.'",
    # "Professor Quirrell, possessed by Voldemort",
    # "Harry's mother's love protected him, causing Quirrell/Voldemort pain when they touched him.",
]


# -----------------------------------------------------------
# Generate Answers and Retrieve Documents for Evaluation Questions
# -----------------------------------------------------------

generated_answers = []        # List to store the generated answers for each question
retrieved_documents = []      # List to store the aggregated context (retrieved documents) for each question

# Iterate over each evaluation question
for question in questions:
    # Prepare the input dictionary for the plan-and-execute pipeline
    input = {"question": question}
    print(f"Answering the question: {question}")

    # Execute the plan-and-execute pipeline and obtain the final answer and state
    final_answer, final_state = execute_plan_and_print_steps(input)

    # Store the generated answer
    generated_answers.append(final_answer)

    # Store the aggregated context (retrieved documents) used to answer the question
    retrieved_documents.append(final_state['aggregated_context'])



# -----------------------------------------------------------
# Display Retrieved Documents and Generated Answers
# -----------------------------------------------------------

# Print the retrieved documents for each evaluation question in a readable format
print(text_wrap(f"retrieved_documents: {retrieved_documents}\n"))

# Print the generated answers for each evaluation question in a readable format
print(text_wrap(f"generated_answers: {generated_answers}"))

# -----------------------------------------------------------
# Prepare Data and Conduct Ragas Evaluation
# -----------------------------------------------------------

# 1. Prepare the data dictionary for Ragas evaluation
data_samples = {
    'question': questions,                # List of evaluation questions
    'answer': generated_answers,          # List of generated answers from the pipeline
    'contexts': retrieved_documents,      # List of aggregated/retrieved contexts for each question
    'ground_truth': ground_truth_answers  # List of ground truth answers for evaluation
}

# 2. Ensure each context is a list of strings (required by Ragas)
#    If each context is a single string, wrap it in a list.
data_samples['contexts'] = [[context] if isinstance(context, str) else context for context in data_samples['contexts']]

# 3. Create a HuggingFace Dataset from the data dictionary
dataset = Dataset.from_dict(data_samples)

# 4. Define the Ragas evaluation metrics to use
# 这部分是定义要用哪些Ragas评测指标（如答案正确性、事实性、相关性、召回率、相似度），
# 后面会用这些指标对生成的答案和检索到的上下文进行自动化评估。
metrics = [
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_similarity
]

# 5. Initialize the LLM for Ragas evaluation (using GPT-4o)
llm = create_llm(4000)

# 6. Run the Ragas evaluation on the dataset with the specified metrics
score = evaluate(dataset, metrics=metrics, llm=llm)

# 7. Convert the results to a pandas DataFrame and print
results_df = score.to_pandas()
print(results_df)


# Call the function to analyze the metric results from the Ragas evaluation
# 'results_df' is the DataFrame containing the evaluation metrics for each question
analyse_metric_results(results_df)  # Analyse the results


