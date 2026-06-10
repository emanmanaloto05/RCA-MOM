from agent_1_sum.chains import (
    summarize_chain,
    extract_owners_chain,
    extract_tasks_chain,
    extract_blockers_chain,
    extract_followups_chain
)

def generate_mom_data(
    transcript: str
):
        summary = summarize_chain(
        transcript
    ).content
        
        owners = extract_owners_chain(
        transcript
    ).content
        
        tasks = extract_tasks_chain(
        transcript
    ).content
        
        blockers = extract_blockers_chain(
        transcript
    ).content
        
        followups = extract_followups_chain(
        transcript
    ).content
    
        return {
        "summary": summary,
        "owners": owners,
        "tasks": tasks,
        "blockers": blockers,
        "followups": followups
    }