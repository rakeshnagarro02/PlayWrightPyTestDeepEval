import os

import pytest
from playwright.sync_api import sync_playwright


def test_example_domain_homepage():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("https://example.com", wait_until="domcontentloaded")

        assert page.title() == "Example Domain"
        assert page.locator("h1").text_content() == "Example Domain"

        browser.close()


def test_json_placeholder_api():
    with sync_playwright() as p:
        api_context = p.request.new_context()
        response = api_context.get("https://jsonplaceholder.typicode.com/todos/1")

        assert response.status == 200
        payload = response.json()
        assert payload["id"] == 1
        assert "title" in payload
        assert payload["completed"] is False


def test_deepeval():
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        from deepeval.models.llms.openai_model import GPTModel
    except ImportError:
        pytest.skip("deepeval is not installed in this environment")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
    if not api_key:
        pytest.skip("Set OPENAI_API_KEY or DEEPEVAL_API_KEY before running this test")

    model_name = os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")
    model = GPTModel(model=model_name, api_key=api_key)
    metric = AnswerRelevancyMetric(threshold=0.7, model=model)

    test_case = LLMTestCase(
        input="What is your return policy?",
        actual_output="We offer a 30-day full refund on all items.",
        retrieval_context=[
            "Our policy allows customers to return products within 30 days for a complete refund."
        ],
    )

    evaluate(test_cases=[test_case], metrics=[metric])

