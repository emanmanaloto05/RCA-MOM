from agent_1_sum.graph import mom_graph


state = {
    "audio_path": "Video Project 3.m4a",
    "transcript_path": "",
    "html_path": "",
    "transcript": "",
    "summary": "",
    "owners": "",
    "tasks": "",
    "blockers": "",
    "followups": "",
    "mom": ""
}


try:
    result = mom_graph.invoke(state)

    print("\n" + "=" * 50)
    print("TRANSCRIPT FILE")
    print("=" * 50)
    print(result["transcript_path"])

    print("\n" + "=" * 50)
    print("HTML FILE")
    print("=" * 50)
    print(result["html_path"])

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(result["summary"])

    print("\n" + "=" * 50)
    print("OWNERS / TASKS / BLOCKERS / FOLLOW-UPS")
    print("=" * 50)
    print(result["owners"])

    print("\n" + "=" * 50)
    print("GENERATED MOM")
    print("=" * 50)
    print(result["mom"])

except Exception as error:
    print("\nERROR:")
    print(error)