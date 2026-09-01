from multi_agent_ops.graph import graph


initial_state = {
    "problem": "Why did customer support resolution time increase?",
    "plan": {},
    "research_findings": {},
    "data_findings": {},
    "business_analysis": {},
    "review": {},
    "status": "RECEIVED",
    "final_report": {},
}


result = graph.invoke(initial_state)

print(result)