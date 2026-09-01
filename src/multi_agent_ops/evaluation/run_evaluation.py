from langsmith import Client
from dotenv import load_dotenv

from multi_agent_ops.graph import build_graph


load_dotenv()


def run_evaluation_dataset():
    """Run the Multi-Agent Ops Crew against the LangSmith evaluation dataset."""

    client = Client()
    graph = build_graph()

    dataset_name = "multi-agent-ops-crew-evaluation"

    datasets = list(
        client.list_datasets(dataset_name=dataset_name)
    )

    if not datasets:
        raise ValueError(
            f"LangSmith dataset '{dataset_name}' was not found."
        )

    dataset = datasets[0]

    print(f"Running dataset: {dataset_name}")
    print()

    examples = list(
        client.list_examples(dataset_id=dataset.id)
    )

    for index, example in enumerate(examples, start=1):
        problem = example.inputs["problem"]

        initial_state = {
            "problem": problem,
            "plan": {},
            "research_findings": {},
            "data_findings": {},
            "business_analysis": {},
            "review": {},
            "status": "STARTED",
            "final_report": {},
        }

        print(f"[{index}/{len(examples)}] {problem}")

        try:
            result = graph.invoke(initial_state)

            print(
                f"Status: {result.get('status')}"
            )

            print(
                f"Final report generated: "
                f"{bool(result.get('final_report'))}"
            )

        except Exception as exc:
            print(f"Evaluation failed: {exc}")

        print("-" * 60)


if __name__ == "__main__":
    run_evaluation_dataset()