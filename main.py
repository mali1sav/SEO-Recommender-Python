import os
import re
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st
import asyncio
import pandas as pd
import hashlib

# Load environment variables from .env file
load_dotenv()

# Constants and Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not set in environment variables.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANALYSIS_MODEL = "anthropic/claude-3.5-sonnet:beta"
INTENT_MODEL = "anthropic/claude-3.5-sonnet:beta"
TITLE_MODEL = "openai/gpt-4o-2024-11-20"
STRUCTURE_MODEL = "anthropic/claude-3.5-sonnet:beta"

# Define Pydantic Models
class KeywordInput(BaseModel):
    raw_input: str = Field(..., description="Raw keyword input in specified format")

class Keyword(BaseModel):
    term: str
    volume: str
    parsed_volume: int

class IntentOutput(BaseModel):
    intent: str

class TitleMetaH1(BaseModel):
    title: str
    metaDescription: str
    h1: str

class Section(BaseModel):
    heading: str
    headingEnglish: str
    targetKeywords: list[str]
    relatedConcepts: list[str]
    contentFormat: str
    details: str
    perplexityLink: str = ""

class SEOPlan(BaseModel):
    title_meta_h1: TitleMetaH1
    content_structure: List[Section]

##############################################################################
# Updated parse_volume function to handle "1.6K" etc.
##############################################################################
def parse_volume(volume_str: str) -> Optional[int]:
    """
    Parses the volume string and converts it to an integer.
    Supports K/M/B suffixes, range formats, and AHREFS-specific formats.
    E.g.: "1.6K" -> 1600, "10K" -> 10000, "0-10" -> 10, "4.5k" -> 4500
    """
    if not volume_str or not isinstance(volume_str, str):
        return None

    volume_str = volume_str.strip().lower().replace(',', '')

    # Handle K, M, B suffix
    # 1.6K -> 1600, 2M -> 2000000, 3.5B -> 3500000000
    match = re.match(r'^(\d+(\.\d+)?)([kmb])$', volume_str)
    if match:
        num_str = match.group(1)  # e.g. 1.6
        suffix = match.group(3)   # e.g. K
        try:
            base_val = float(num_str)
            if suffix == 'k':
                return int(base_val * 1_000)
            elif suffix == 'm':
                return int(base_val * 1_000_000)
            elif suffix == 'b':
                return int(base_val * 1_000_000_000)
        except ValueError:
            return None

    # Handle range formats like "0-10"
    if '-' in volume_str:
        try:
            start, end = map(str.strip, volume_str.split('-'))
            # Take the higher number
            return int(float(end))
        except ValueError:
            pass

    # If no suffix or range, try direct integer or float
    try:
        return int(float(volume_str))
    except ValueError:
        pass

    return None

##############################################################################
# Updated parse_keywords to call parse_volume instead of stripping all non-digits
##############################################################################
def parse_keywords(raw_input: str) -> List[Keyword]:
    """
    Parses the raw keyword input into a list of Keyword objects.
    Expects keywords and volumes on alternating lines, e.g.:
      line 1: "leverage คือ"
      line 2: "1.6K"
      line 3: "leverage ratio คือ"
      line 4: "200"
    """
    keywords = []
    lines = [line.strip() for line in raw_input.split('\n') if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            term = lines[i]
            volume_str = lines[i + 1]
            parsed_vol = parse_volume(volume_str)
            if parsed_vol is None:
                parsed_vol = 0  # fallback if parsing fails
            if term and volume_str:
                keywords.append(Keyword(
                    term=term,
                    volume=volume_str,
                    parsed_volume=parsed_vol
                ))
    return keywords

def create_session():
    """
    Creates a requests session with retry logic.
    """
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Create the session early
session = create_session()

async def call_openrouter(prompt: str, model: str) -> str:
    """
    Calls the OpenRouter API with the given prompt and model.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/your-username/seo-recommender",  # Replace with your actual URL
        "X-Title": "SEO Content Recommender"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    for attempt in range(3):
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: session.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=data
                )
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                raise ValueError("Invalid response format from OpenRouter.")
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                raise Exception(f"OpenRouter API call failed: {str(e)}")

async def generate_intent(keywords: List[Keyword]) -> IntentOutput:
    """
    Generates content intent based on the list of keywords.
    """
    if not keywords:
        raise Exception("Keyword list is empty.")
    keywords_list = "\n".join([kw.term for kw in keywords])
    prompt = f"""
    You are an SEO expert analyzing a list of Thai keywords to suggest a comprehensive content intent. The intent should cover all major aspects indicated by the keywords while maintaining natural language flow.

    Keywords to analyze:
    {keywords_list}

    Based on these keywords, generate a content intent in Thai that:
    1. Captures the main topic and its key aspects
    2. Addresses the implied user questions and search intent
    3. Provides a logical flow for content structure
    4. Uses natural Thai language
    5. Is comprehensive but concise (around 1-2 sentences)

    Return ONLY the suggested intent in Thai, with no additional text or explanation.

    Example format of the response:
    ข้อมูล [topic] ใน[context], [aspect1], [aspect2], และ[aspect3]
    """
    intent = await call_openrouter(prompt, INTENT_MODEL)
    if not intent:
        raise Exception("Failed to generate intent.")
    return IntentOutput(intent=intent)

async def analyze_keyword_relevancy(keywords: List[Keyword]) -> List[Dict]:
    """
    Analyzes keywords and returns relevancy scores with descriptions in Thai.
    """
    keywords_list = "\n".join([f"{kw.term} ({kw.parsed_volume})" for kw in keywords])
    prompt = f"""
    Analyze the relevancy of ALL the following keywords:
    {keywords_list}

    For each keyword, return a JSON array with ALL keywords analyzed. Each object must have:
    1. keyword: [keyword]
    2. volume: [search volume] (integer)
    3. relevancy: relevancy score (integer 0-100)
    4. description: less than 50 characters description explaining why you gave the relevancy score

    Rules for relevancy scoring:
    - Main definition keywords (with คือ) should get 90-100%
    - Related application/usage keywords should get 70-90%
    - Peripheral topics should get 40-70%
    - Loosely related topics should get below 40%

    Example format of the response:
    [
        {{
            "keyword": "fibonacci คือ",
            "volume": 700,
            "relevancy": 100,
            "description": "คำนิยามหลักของ Fibonacci ซึ่งเป็นหัวข้อหลัก"
        }}
    ]

    Return ALL keywords in the JSON array. Do not skip any keywords.
    Only return JSON array, no additional text or explanation.
    """
    try:
        result = await call_openrouter(prompt, ANALYSIS_MODEL)
        print(f"API Response: {result}")  # Debug

        result = result.strip()
        # If response starts with ```json, remove it
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '').strip()

        analysis = json.loads(result)

        # Validate structure
        for item in analysis:
            required_keys = ['keyword', 'volume', 'relevancy', 'description']
            if not all(key in item for key in required_keys):
                raise ValueError(f"Missing required keys in item: {item}")
            if not isinstance(item['relevancy'], (int, float)) or not 0 <= item['relevancy'] <= 100:
                raise ValueError(f"Invalid relevancy value in item: {item}")

        return analysis
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response as JSON. Response: {result[:200]}...") from e
    except ValueError as e:
        raise Exception(f"Invalid response format: {str(e)}") from e
    except Exception as e:
        raise Exception(f"Error analyzing keywords: {str(e)}") from e

async def full_seo_plan(keyword_input: KeywordInput) -> SEOPlan:
    """
    Complete workflow to generate SEO plan from raw keyword input.
    """
    # Step 1: Process Keywords with volumes
    keywords = []
    for line in keyword_input.raw_input.split('\n'):
        if '|' in line:
            term, volume = line.split('|')
            keywords.append({"term": term.strip(), "volume": int(volume.strip())})

    if not keywords:
        raise Exception("No valid keywords found in input. Make sure keywords are in format: term|volume")

    # Get main keyword (first in the list)
    main_keyword = keywords[0]["term"]

    # Step 2: Generate Title, Meta, and H1 with focus on main keyword
    title_prompt = f"""
    Based on this main keyword: {main_keyword}
    And these supporting keywords:
    {', '.join([f"{kw['term']} ({kw['volume']})" for kw in keywords[1:]])}

    Generate a title, meta description, and H1 in Thai that:
    1. Naturally integrates the main keyword early in each element
    2. Uses supporting keywords where natural
    3. Is compelling, engaging and click-worthy
    4. Stays within character limits (title: 60 chars, meta: 155 chars)

    Return in this exact JSON format:
    {{
        "title": "Your SEO title here",
        "metaDescription": "Your meta description here",
        "h1": "Your H1 here"
    }}

    Return ONLY the JSON, no other text.
    """
    title_response = await call_openrouter(title_prompt, TITLE_MODEL)
    try:
        title_response = title_response.strip()
        if title_response.startswith('```json'):
            title_response = title_response.replace('```json', '').replace('```', '').strip()

        title_meta_h1 = json.loads(title_response)

        required_keys = ["title", "metaDescription", "h1"]
        if not all(k in title_meta_h1 for k in required_keys):
            raise ValueError(f"Missing required keys. Got: {list(title_meta_h1.keys())}")

        for k in required_keys:
            if not isinstance(title_meta_h1[k], str) or not title_meta_h1[k].strip():
                raise ValueError(f"Invalid or empty value for {k}")

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse title/meta/H1 response as JSON. Response: {title_response[:200]}...") from e
    except ValueError as e:
        raise Exception(f"Invalid title/meta/H1 format: {str(e)}") from e
    except Exception as e:
        raise Exception(f"Error generating title/meta/H1: {str(e)}") from e

    # Step 3: Generate Content Structure
    content_structure_prompt = f"""
    Create a detailed content structure in Thai for an article with main keyword: {main_keyword}

    Supporting keywords:
    {', '.join([f"{kw['term']} ({kw['volume']})" for kw in keywords[1:]])}

    Title: {title_meta_h1['title']}
    Intent: {st.session_state.content_intent}

    Create 3-6 distinct sections (including introduction) that progress logically, making it easy to navigate. Each section should:
    1. Have a clear, distinct purpose that aligns with the main intent
    2. Group related keywords together based on user search intent
    3. Progress logically from:
       - Introduction & Basic Definitions
       - Core Concepts & Main Information
       - Detailed Analysis & Advanced Topics

    For each section:
    1. Create an H2 heading in Thai that:
       - Naturally incorporates relevant keywords without forcing them
       - Avoids redundant or repetitive terms
       - If incorporating multiple keywords, expands into a natural sentence
       - Flows logically from the previous heading
    2. Provide the English translation of the H2 for Perplexity search
    3. List target keywords with their search volumes that belong in this section
    4. Combine reasoning, context, and guidelines into a single 'Details' field explaining the section's purpose and how to handle it
    5. List 3-5 related concepts to cover
    6. Suggest content format

    Return in this exact JSON format:
    [
        {{
            "heading": "Thai H2 heading",
            "headingEnglish": "English translation of H2",
            "targetKeywords": ["keyword1 (volume1)", "keyword2 (volume2)"],
            "relatedConcepts": ["concept1", "concept2", "concept3"],
            "contentFormat": "Format suggestion i.e. paragraphs, list, table, step by step, etc.",
            "details": "Combined reasoning, context, and guidelines in Thai",
            "perplexityLink": "https://www.perplexity.ai/search?q=..."
        }},
        ...
    ]

    Return ONLY the JSON array, no other text.
    """
    structure_response = await call_openrouter(content_structure_prompt, STRUCTURE_MODEL)
    try:
        structure_response = structure_response.strip()
        if structure_response.startswith('```json'):
            structure_response = structure_response.replace('```json', '').replace('```', '').strip()

        content_structure = json.loads(structure_response)
        if not isinstance(content_structure, list):
            raise ValueError("Content structure must be a list")

        required_keys = ["heading", "headingEnglish", "targetKeywords", "relatedConcepts", "contentFormat", "details"]
        for section in content_structure:
            if not all(key in section for key in required_keys):
                raise ValueError(f"Missing required keys in section: {list(section.keys())}")

            section["perplexityLink"] = f"https://www.perplexity.ai/search?q={'+'.join(section['headingEnglish'].split())}+.+Explain+in+Thai"

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse content structure response as JSON. Response: {structure_response[:200]}...") from e
    except ValueError as e:
        raise Exception(f"Invalid content structure format: {str(e)}") from e
    except Exception as e:
        raise Exception(f"Error generating content structure: {str(e)}") from e

    return SEOPlan(
        title_meta_h1=TitleMetaH1(**title_meta_h1),
        content_structure=[Section(**section) for section in content_structure]
    )

# Streamlit UI
st.set_page_config(page_title="SEO Content Recommender", layout="wide")
st.title("SEO Content Recommender")

# Initialize session state variables if they don't exist
if 'analyzed_keywords' not in st.session_state:
    st.session_state.analyzed_keywords = []
if 'keyword_selections' not in st.session_state:
    st.session_state.keyword_selections = {}
if 'content_intent' not in st.session_state:
    st.session_state.content_intent = None
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'intent_generated' not in st.session_state:
    st.session_state.intent_generated = False

def display_analyzed_keywords():
    if 'analyzed_keywords' not in st.session_state:
        return 0

    total_volume = 0
    
    st.markdown(f"### Total Keywords: {len(st.session_state.analyzed_keywords)}")
    st.markdown("Keywords with relevancy ≥ 60% are selected by default")
    
    cols = st.columns([0.1, 0.5, 0.2, 0.2])
    cols[0].markdown("**Select**")
    cols[1].markdown("**Keyword**")
    cols[2].markdown("**Volume**")
    cols[3].markdown("**Relevancy**")
    
    for i, kw in enumerate(st.session_state.analyzed_keywords):
        cols = st.columns([0.1, 0.5, 0.2, 0.2])
        unique_key = f"keyword_select_{i}"
        
        is_selected = kw["relevancy"] >= 60
        selected = cols[0].checkbox("", key=unique_key, value=is_selected)
        st.session_state.keyword_selections[unique_key] = selected
        
        cols[1].markdown(f"{kw['keyword']}")
        cols[2].markdown(f"{kw['volume']:,}")
        
        relevancy_color = "green" if kw["relevancy"] >= 60 else "red"
        cols[3].markdown(f"<span style='color:{relevancy_color}'>{kw['relevancy']}%</span>", unsafe_allow_html=True)
        
        if selected:
            total_volume += kw['volume']
    
    st.markdown("---")
    st.markdown(f"**Total Monthly Search Volume**: {total_volume:,}")
    return total_volume

raw_input = st.text_area("Enter your keywords:", height=200)

if st.button("Suggest Intent"):
    if raw_input:
        try:
            with st.spinner("Analyzing keywords and generating intent..."):
                keyword_input = KeywordInput(raw_input=raw_input)
                keywords = parse_keywords(raw_input)

                intent_output = asyncio.run(generate_intent(keywords))

                st.session_state.keywords = keywords
                st.session_state.content_intent = intent_output.intent
                st.session_state.intent_generated = True

                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some keywords first.")

if st.session_state.intent_generated:
    st.subheader("Content Intent (Thai)")
    edited_intent = st.text_area(
        "You can edit this suggested intent:",
        value=st.session_state.content_intent,
        height=100,
        key="intent_editor"
    )
    st.session_state.content_intent = edited_intent

if st.session_state.intent_generated:
    if st.button("Analyze Keyword Relevancy"):
        try:
            with st.spinner("Analyzing keyword relevancy..."):
                if not st.session_state.analyzed_keywords:
                    analyzed = asyncio.run(analyze_keyword_relevancy(st.session_state.keywords))
                    st.session_state.analyzed_keywords = analyzed
                    st.session_state.keyword_selections = {}
                    for i, kw in enumerate(analyzed):
                        unique_key = f"keyword_select_{i}"
                        st.session_state.keyword_selections[unique_key] = kw['relevancy'] >= 50
        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.session_state.analyzed_keywords:
    display_analyzed_keywords()

if st.session_state.intent_generated and st.session_state.analyzed_keywords:
    if st.button("Generate Full SEO Plan"):
        try:
            with st.spinner("Generating SEO plan..."):
                selected_keywords = []
                for i, kw in enumerate(st.session_state.analyzed_keywords):
                    key_used = f"keyword_select_{i}"
                    if st.session_state.keyword_selections.get(key_used, False):
                        selected_keywords.append(kw)

                if not selected_keywords:
                    st.error("Please select at least one keyword")
                    st.stop()

                st.markdown("### Selected Keywords for SEO Plan")
                table_data = {
                    "Keyword": [kw['keyword'] for kw in selected_keywords],
                    "Monthly Volume": [kw['volume'] for kw in selected_keywords]
                }
                df = pd.DataFrame(table_data)
                st.table(df)

                total_volume = sum(kw['volume'] for kw in selected_keywords)
                st.markdown(f"**Total Monthly Search Volume**: {total_volume:,}")
                st.markdown("---")

                keyword_input_str = "\n".join([
                    f"{kw['keyword']}|{kw['volume']}"
                    for kw in selected_keywords
                ])

                seo_plan = asyncio.run(full_seo_plan(KeywordInput(raw_input=keyword_input_str)))

                st.markdown("## SEO Content Plan")
                st.markdown(f"**Title**: {seo_plan.title_meta_h1.title}")
                st.markdown(f"**Meta Description**: {seo_plan.title_meta_h1.metaDescription}")
                st.markdown(f"**H1**: {seo_plan.title_meta_h1.h1}")

                st.markdown("## Content Structure")
                for section in seo_plan.content_structure:
                    heading = re.sub(r'\s*\(\d+\)\s*', '', section.heading)
                    st.markdown(f"## **{heading}**")
                    st.markdown("**Target Keywords**: " + ", ".join(section.targetKeywords))
                    st.markdown("**Related Concepts**: " + ", ".join(section.relatedConcepts))
                    st.markdown(f"**Content Format**: {section.contentFormat}")
                    st.markdown(f"**Guidelines**: {section.details}")
                    st.markdown(f"**Research**: [{section.perplexityLink}]({section.perplexityLink})")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error: {str(e)}")
