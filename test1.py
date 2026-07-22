import os

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.llms.openai_model import GPTModel

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
if not api_key:
    raise RuntimeError("Set the OPENAI_API_KEY or DEEPEVAL_API_KEY environment variable before running this script.")

model_name = os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")
model = GPTModel(model=model_name, api_key=api_key)
metric = AnswerRelevancyMetric(threshold=0.7, model=model)

# 1. Define your test case
test_case = LLMTestCase(
    input="What is your return policy?",
    actual_output="We offer a 30-day full refund on all items.",
    retrieval_context=["Our policy allows customers to return products within 30 days for a complete refund."]
)


# 3. Run evaluation directly in Python
evaluate(test_cases=[test_case], metrics=[metric])