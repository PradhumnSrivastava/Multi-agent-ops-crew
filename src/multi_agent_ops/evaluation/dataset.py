from dotenv import load_dotenv
from langsmith import Client


load_dotenv()


DATASET_NAME = "multi-agent-ops-crew-evaluation"


EVALUATION_EXAMPLES = [
    {
        "problem": "Why did customer support resolution time increase?",
        "expected_focus": [
            "increase in ticket volume",
            "reduction in staff count",
            "increase in workload per staff member",
            "increase in average resolution time",
        ],
    },
    {
        "problem": "Why did operational costs increase?",
        "expected_focus": [
            "increase in operational expenses",
            "changes in staffing or resource costs",
            "increase in workload or activity",
            "potential operational inefficiencies",
        ],
    },
    {
        "problem": "Why did employee productivity decrease?",
        "expected_focus": [
            "change in employee productivity",
            "staffing changes",
            "workload changes",
            "potential operational factors affecting productivity",
        ],
    },
    {
        "problem": "Why did customer complaints increase?",
        "expected_focus": [
            "increase in complaint volume",
            "customer support performance",
            "resolution time or service quality",
            "potential contributing operational factors",
        ],
    },
    {
        "problem": "Why did customer support workload increase?",
        "expected_focus": [
            "increase in support tickets",
            "change in staffing",
            "workload per staff member",
            "potential drivers of increased support demand",
        ],
    },
]


def get_or_create_dataset(client: Client):
    """Get the evaluation dataset or create it if necessary."""

    datasets = list(
        client.list_datasets(
            dataset_name=DATASET_NAME
        )
    )

    if datasets:
        return datasets[0]

    return client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Evaluation dataset for the Multi-Agent Ops Crew. "
            "Contains business operations problems and "
            "reference criteria for evaluating generated reports."
        ),
    )


def update_examples(
    client: Client,
    dataset_id,
) -> None:
    """Add reference outputs to existing evaluation examples."""

    examples = list(
        client.list_examples(
            dataset_id=dataset_id
        )
    )

    examples_by_problem = {
        example.inputs.get("problem"): example
        for example in examples
    }

    updated = 0
    created = 0

    for evaluation_example in EVALUATION_EXAMPLES:
        problem = evaluation_example["problem"]

        reference_output = {
            "expected_focus": evaluation_example[
                "expected_focus"
            ]
        }

        existing_example = examples_by_problem.get(
            problem
        )

        if existing_example:
            client.update_example(
                existing_example.id,
                outputs=reference_output,
            )
            updated += 1

        else:
            client.create_example(
                inputs={
                    "problem": problem
                },
                outputs=reference_output,
                dataset_id=dataset_id,
            )
            created += 1

    print(
        f"Updated examples: {updated}"
    )
    print(
        f"Created missing examples: {created}"
    )


def create_dataset() -> None:
    """Prepare the LangSmith evaluation dataset."""

    client = Client()

    dataset = get_or_create_dataset(
        client
    )

    print(
        f"Dataset: {dataset.name}"
    )

    print(
        f"Dataset ID: {dataset.id}"
    )

    update_examples(
        client,
        dataset.id,
    )

    examples = list(
        client.list_examples(
            dataset_id=dataset.id
        )
    )

    print(
        f"Total examples: {len(examples)}"
    )

    print(
        "\nDataset preparation completed."
    )


if __name__ == "__main__":
    create_dataset()