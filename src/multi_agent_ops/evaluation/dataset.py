from langsmith import Client
from dotenv import load_dotenv
import os


load_dotenv()


def create_evaluation_dataset():
    """Create the evaluation dataset for Multi-Agent Ops Crew."""

    client = Client()

    dataset_name = "multi-agent-ops-crew-evaluation"

    examples = [
        {
            "problem": "Why did customer support resolution time increase?",
        },
        {
            "problem": "Why did customer support workload increase?",
        },
        {
            "problem": "Why did operational costs increase?",
        },
        {
            "problem": "Why did customer complaints increase?",
        },
        {
            "problem": "Why did employee productivity decrease?",
        },
    ]

    existing_datasets = list(
        client.list_datasets(dataset_name=dataset_name)
    )

    if existing_datasets:
        print(f"Dataset already exists: {dataset_name}")
        return

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Evaluation dataset for testing the Multi-Agent Ops Crew "
            "business analysis workflow."
        ),
    )

    client.create_examples(
        inputs=examples,
        dataset_id=dataset.id,
    )

    print(f"Created dataset: {dataset_name}")
    print(f"Examples: {len(examples)}")


if __name__ == "__main__":
    create_evaluation_dataset()