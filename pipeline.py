from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}
    
    print("\n[1/4] Running Search Agent...")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    }) # type: ignore
    
    raw_search = search_result["messages"][-1].content 
    

    if isinstance(raw_search, list):
        clean_search_text = ""
        for block in raw_search:
            if isinstance(block, dict) and block.get("type") == "text":
                clean_search_text += block.get("text", "") + "\n"
        state["search_results"] = clean_search_text.strip()
    else:
        state["search_results"] = str(raw_search)
        
    print("-> Search completed successfully.")
    
    print("\n[2/4] Running Reader Agent...")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
                       f"Based on the following search results about {topic},"
                       f"pick the most relevant URL and scrape it for deeper content,\n\n"
                       f"Search Results:\n{state['search_results'][:2000]}")]
    }) # type: ignore
    
    raw_scraped = reader_result['messages'][-1].content
    
    if isinstance(raw_scraped, list):
        clean_scraped_text = ""
        for block in raw_scraped:
            if isinstance(block, dict) and block.get("type") == "text":
                clean_scraped_text += block.get("text", "") + "\n"
        state['scrapped_content'] = clean_scraped_text.strip()
    else:
        state['scrapped_content'] = str(raw_scraped)

    print("-> Scraping completed successfully.")
    safe_scraped_content = state['scrapped_content'][:10000]
    
    print("\n[3/4 & 4/4] Starting Writer/Critic Iteration Loop...")
    
    max_revisions = 3
    revision_count = 0
    current_draft = "None yet."
    critic_feedback = "None yet. Please write the initial draft."
    is_approved = False

    while revision_count <= max_revisions:
        print(f"\n--- Iteration {revision_count + 1} ---")
        
        # 1. Writer creates or revises the draft
        print("-> Writer is drafting/revising...")
        writer_output = writer_chain.invoke({
            "topic": topic,
            "raw_market_data": state["search_results"][:5000],
            "raw_competitor_data": safe_scraped_content,
            "previous_draft": current_draft,
            "critic_feedback": critic_feedback,
            "messages": [] 
        })
        current_draft = writer_output.content if hasattr(writer_output, 'content') else str(writer_output)
        
        # 2. Critic reviews the draft
        print("-> Critic is evaluating the draft...")
        critic_output = critic_chain.invoke({
            "topic": topic,
            "raw_market_data": state["search_results"][:5000],
            "raw_competitor_data": safe_scraped_content,
            "current_draft": current_draft
        }) 
        critic_feedback = critic_output.content if hasattr(critic_output, 'content') else str(critic_output)
        
        # 3. Check if the Critic approved it
        if "APPROVE" in critic_feedback.upper():
            print("\n Critic APPROVED the draft!")
            is_approved = True
            break
        else:
            print("\n Critic REJECTED the draft. Sending back to Writer...")
            revision_count += 1

    if not is_approved:
        print(f"\n Reached maximum revisions ({max_revisions}). Proceeding with final draft.")

    state["report"] = current_draft
    state["feedback"] = critic_feedback
    
    return state

if __name__ == "__main__":
    topic = input("\nEnter a topic: ")
    output_state = run_research_pipeline(topic)
    
    print("\n" + "="*40 + " FINAL REPORT " + "="*40)
    print(output_state["report"])
    print("\n" + "="*40 + " PARTNER FEEDBACK " + "="*40)
    print(output_state["feedback"])