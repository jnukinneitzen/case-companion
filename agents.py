import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda  
from tools import web_search, scrape_url

load_dotenv()

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.1-8b-instant", 
    temperature=0.0,
    max_retries=5
)


def rate_limit_pacing(inputs):
    """Introduces an intentional cooling period to stay within API token/request limits."""
    time.sleep(2.5) 
    return inputs

pacing_layer = RunnableLambda(rate_limit_pacing)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )



def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )



writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Senior Engagement Manager at a top-tier management consulting firm.
Your objective is to synthesize raw market research and competitor data into a comprehensive, boardroom-ready strategy document.

CRITICAL INSTRUCTIONS:
1. FACTUAL RIGOR: Base all claims ONLY on the provided data. Do not hallucinate metrics. Cite sources inline as [Source: URL].
2. PROFESSIONAL TONE: Use razor-sharp, executive-level language. No fluff, no introductory filler. 
3. NATIVE VISUAL CHARTS: Since you cannot render standard graphic image files, you MUST use Markdown tables containing structural Unicode bar blocks (e.g., '█', '▌') to construct high-impact, scannable text-based bar charts directly inside your performance logs and tables. Normalize data to a clear scale (e.g., 1 block = 10% or 10 units) so comparative metrics are immediately intuitive. For high-priority metrics, follow the visual with a short bracketed tag indicating slide creation intent, like: [SLIDE CHART: Bar Chart showing X vs Y].

REQUIRED REPORT STRUCTURE:
You must strictly follow this heading hierarchy and complete every section:
# 1. Executive Summary
# 2. Market Sizing & Landscape
# 3. Macro-Environmental Assessment (PESTLE)
# 4. Competitor Analysis & Positioning
# 5. Strategic Viability & Operational Feasibility
# 6. Risk Matrix & Mitigation"""
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "human",
        """Target Topic/Company: {topic}

--- RAW MARKET DATA ---
{raw_market_data}

--- RAW COMPETITOR DATA ---
{raw_competitor_data}

--- PREVIOUS DRAFT (If revising) ---
{previous_draft}

--- CRITIC FEEDBACK (If revising) ---
{critic_feedback}

INSTRUCTIONS: 
If 'CRITIC FEEDBACK' is provided, you MUST revise the 'PREVIOUS DRAFT' to address every single point the critic made. Do not ignore the feedback.
If no feedback is provided, write the first draft from scratch."""
    )
])

writer_chain = pacing_layer | writer_prompt | llm | StrOutputParser()


critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a rigorous Senior Partner at a top-tier management consulting firm. 
Your Engagement Manager (The Writer) has submitted a draft of a strategy report. Your job is to review it against the raw data to ensure it is boardroom-ready.

YOUR EVALUATION CRITERIA:
1. Data Grounding: Did the Writer hallucinate any numbers? Are all metrics backed by the raw data?
2. Framework Integrity: Are the PESTLE, SWOT, and Porter's 5 Forces complete and logically sound?
3. Tone, Formatting & Visuals: Is the tone sharp and executive? Did the Writer accurately generate clean Markdown data tables embedded with functional, proportionate Unicode block bar charts to visualize numerical patterns seamlessly?

If the draft fails ANY of these criteria, you must REJECT it and provide a strict bulleted list of fixes.
If the draft is flawless, factually accurate, and well-structured, you may APPROVE it."""
    ),
    (
        "human",
        """Target Topic: {topic}

--- RAW DATA SOURCED ---
Market Data: {raw_market_data}
Competitor Data: {raw_competitor_data}

--- DRAFT FOR REVIEW ---
{current_draft}

Evaluate the draft and provide your decision and feedback."""
    )
])

critic_chain = pacing_layer | critic_prompt | llm | StrOutputParser()