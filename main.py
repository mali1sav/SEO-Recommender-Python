import os
import re
import json
import time
from urllib.parse import quote_plus  # สำหรับการเข้ารหัส URL

import requests
from requests.adapters import Retry, HTTPAdapter
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client
def init_gemini_client():
    """Initialize Google Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        return {
            'model': model,
            'name': 'gemini-2.0-flash-exp'
        }
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        return None

# Initialize Gemini client
gemini_client = init_gemini_client()
if not gemini_client:
    raise EnvironmentError("Failed to initialize Gemini client.")

# Define Pydantic Models
class KeywordInput(BaseModel):
    raw_input: str = Field(..., description="Raw keyword input in specified format")

class Keyword(BaseModel):
    term: str
    volume: str
    parsed_volume: int

class IntentOutput(BaseModel):
    intents: List[str]
    selected_intent: Optional[str] = None

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
    details: Union[str, List[Dict[str, str]]]  # Allow both string and list of dictionaries
    perplexityLink: str = ""

class SEOPlan(BaseModel):
    title_meta_h1: TitleMetaH1
    content_structure: List[Section]

# Utility Functions
def parse_keywords(raw_input: str) -> List[Keyword]:
    """
    Parses the raw keyword input into a list of Keyword objects.
    Expects keywords and volumes on alternating lines.
    Includes all keywords, even if the volume is None.
    """
    keywords = []
    lines = [line.strip() for line in raw_input.split('\n') if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            term = lines[i]
            volume_str = lines[i + 1]
            
            # Parse the volume
            parsed_volume = parse_volume(volume_str)
            
            # Include the keyword even if the volume is None
            keywords.append(Keyword(
                term=term,
                volume=volume_str,  # เก็บค่า volume แบบเดิม
                parsed_volume=parsed_volume if parsed_volume is not None else 0  # กำหนดเป็น 0 ถ้าไม่ parse ได้
            ))
    
    return keywords

def parse_volume(volume_str: str) -> Optional[int]:
    """
    Parses the volume string and converts it to an integer.
    Supports K/M/B suffixes, range formats, and AHREFS-specific formats.
    Treats "0-10" and "Keyword not indexed in Ahrefs database" as 10.
    Returns None if parsing fails.
    """
    if not volume_str or not isinstance(volume_str, str):
        return None

    volume_str = volume_str.strip().lower()

    if volume_str == "0-10" or volume_str == "keyword not indexed in ahrefs database":
        return 10

    if 'k' in volume_str:
        try:
            num = float(volume_str.replace('k', '').replace(',', ''))
            return int(num * 1000)
        except ValueError:
            pass

    if '-' in volume_str:
        try:
            start, end = map(str.strip, volume_str.split('-'))
            return int(end)
        except ValueError:
            pass

    try:
        return int(volume_str.replace(',', ''))
    except ValueError:
        pass

    return None

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

def extract_promotional_entities(intent: str) -> List[str]:
    """
    Rotates promotional entities with brand rotation list.
    """
    idx = int(time.time()) % len(BRAND_ROTATION)
    return BRAND_ROTATION[idx:idx+2]

session = create_session()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=20),
    retry=tenacity.retry_if_exception_type((Exception)),
    retry_error_callback=lambda retry_state: None
)
def make_gemini_request(prompt: str) -> str:
    """Make Gemini API request with retries and proper error handling"""
    try:
        for attempt in range(3):
            try:
                response = gemini_client['model'].generate_content(prompt)
                if response and response.text:
                    return response.text
            except Exception as e:
                if attempt == 2:
                    raise
                st.warning(f"Retrying Gemini request (attempt {attempt + 2}/3)...")
                time.sleep(2 ** attempt)
        raise Exception("Failed to get valid response from Gemini API after all retries")
    except Exception as e:
        st.error(f"Error making Gemini request: {str(e)}")
        raise

CONTENT_TYPES = [
    "Listicle", 
    "How-To Guide",
    "Comparison Article",
    "Expert Insights",
    "Problem-Solving"
]

CONTENT_STRUCTURES = [
    "Table-based comparison",
    "Step-by-step guide",
    "Checklist format", 
    "Case Study/Storytelling",
    "Q&A format"
]

BRAND_ROTATION = [
    "BestWallet.com/th",
    "MetaMask",
    "Trust Wallet",
    "Binance Wallet",
    "Coinbase Wallet"
]

def rotate_keywords(keywords: List[Keyword], num_groups=5) -> List[List[str]]:
    """Split keywords into rotating groups"""
    return [keywords[i::num_groups] for i in range(num_groups)]

def generate_intent(keywords: List[Keyword]) -> IntentOutput:
    """
    Generates content intent with enforced content types.
    """
    if not keywords:
        raise Exception("Keyword list is empty.")
        
    keywords_list = "\n".join([kw.term for kw in keywords])
    prompt = f"""
    Generate 5 distinct Thai content intents. Feel free to use these example formats or better ones:

    1. Listicle: "10 ข้อควรรู้เกี่ยวกับ [main keyword]" 
    2. How-To: "วิธี [action] [main keyword] แบบมืออาชีพ"
    3. Comparison: "[option1] vs [option2] ต่างกันอย่างไร?"
    4. Expert: "ผู้เชี่ยวชาญเผย [insight] เกี่ยวกับ [main keyword]"
    5. Problem-Solving: "แก้ไขปัญหา [common issue] ด้วย [solution]"

    Keywords: {keywords_list}

    Return EXACTLY 5 intents matching these formats in Thai:
    1. Captures the main topic from a unique angle or perspective
    2. Addresses different implied user questions and search intents
    3. Provides a logical flow for content structure
    4. Uses natural Thai language while retaining SEO keywords in their original form, whether they are in Thai or English
    5. Is comprehensive but concise (around 1-2 sentences)

    Return EXACTLY 5 intents in Thai, one per line, numbered from 1-5. No additional text or explanation.

    Example format of the response:
    1. ข้อมูล [topic] ใน[context], [aspect1], [aspect2], และ[aspect3]
    2. วิธีการ [topic] สำหรับ[target], [benefit1], [benefit2], และ[benefit3]
    3. เปรียบเทียบ [topic] ระหว่าง[option1], [option2], และ[option3]
    4. แนวทางการ [topic] ที่[characteristic], [feature1], [feature2]
    5. ความสำคัญของ [topic] ต่อ[impact1], [impact2], และ[impact3]
    """
    intent_response = make_gemini_request(prompt)
    if not intent_response:
        raise Exception("Failed to generate intents.")
    
    intents = [line.strip() for line in intent_response.split('\n') if line.strip()]
    intents = [re.sub(r'^\d+\.\s*', '', intent) for intent in intents]
    
    return IntentOutput(intents=intents)

def analyze_keyword_relevancy(keywords: List[Keyword]) -> List[Dict]:
    """
    Analyzes keywords and returns relevancy scores with descriptions in Thai.
    """
    keywords_list = "\n".join([f"{kw.term} ({kw.parsed_volume})" for kw in keywords])
    prompt = f"""
    Analyze the relevancy of ALL the following keywords based on the search intent:
    Search Intent: {st.session_state.content_intent}  # <-- Pass the intent here

    Keywords to analyze:
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
        result = make_gemini_request(prompt)
        print(f"API Response: {result}")

        result = result.strip()
        if result.startswith('```json'):
            result = result.replace('```json', '').replace('```', '').strip()

        analysis = json.loads(result)

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

def full_seo_plan(keyword_input: KeywordInput) -> SEOPlan:
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

    main_keyword = keywords[0]["term"]

    # Step 2: Detect promotional entities
    promotional_entities = extract_promotional_entities(st.session_state.content_intent)

    # Step 3: Generate Title, Meta, and H1 with focus on main keyword
    title_prompt = f"""
    Based on this main keyword: {main_keyword}
    And these supporting keywords:
    {', '.join([f"{kw['term']} ({kw['volume']})" for kw in keywords[1:]])}

    Search Intent: {st.session_state.content_intent}  # <-- Pass the intent here

    Generate a title, meta description, and H1 in Thai that:
    1. Naturally integrates {main_keyword} early in each element
    2. Uses supporting keywords where natural
    3. Is compelling, engaging and click-worthy
    4. Stays within character limits (title: 60 chars, meta: 155 chars)
    5. Does NOT include promotional content like "{', '.join(promotional_entities)}" unless it is part of the main topic.

    Return in this exact JSON format:
    {{
        "title": "Your SEO title here",
        "metaDescription": "Your meta description here",
        "h1": "Your H1 here"
    }}

    Return ONLY the JSON, no other text.
    """
    title_response = make_gemini_request(title_prompt)
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

    # Step 4: Generate Content Structure
    # เงื่อนไขสำหรับการรวมคำแนะนำเกี่ยวกับ meme coin เฉพาะเมื่อ main keyword เกี่ยวกับ meme
    recent_insights_guideline = ""
    if "meme" in main_keyword.lower():
        recent_insights_guideline = """
    5. Optionally, if it enhances the article’s relevance, include recent insights such as:
       - Trump launching a meme coin.
       - Elon Musk indirectly endorsing DOGE (for example, by naming his new Department of Government Efficiency DOGE).
       - David Sacks (the new Crypto czar) stating that NFTs and meme coins are collectible.
        """
    content_structure_prompt = f"""
    Create a detailed content structure in Thai for an article with main keyword: {main_keyword}

    Supporting keywords:
    {', '.join([f"{kw['term']} ({kw['volume']})" for kw in keywords[1:]])}

    Title: {title_meta_h1['title']}
    Search Intent: {st.session_state.content_intent}  # <-- Pass the intent here

    Create **4-6 distinct sections** (including introduction and FAQs) that progress logically, making it easy to navigate. Each section should:
    1. Have a clear, distinct purpose that aligns with the search intent: "{st.session_state.content_intent}"
    2. Group related keywords together based on the search intent
    3. Progress logically from:
       - Introduction & Basic Definitions
       - Core Concepts & Main Information
       - Detailed Analysis & Advanced Topics
       - Related Subtopics and Additional Insights
    4. If the search intent mentions any promotional entities (e.g., {', '.join(promotional_entities)}), integrate them naturally into the content sections without being overly promotional. Specifically:
       - Mention the promotional entity (e.g., {', '.join(promotional_entities)}) in at least one section's guidelines or details.
       - Ensure the mention feels natural and adds value to the content.
    {recent_insights_guideline}
    For each section (except FAQs):
    1. Create an H2 heading in Thai that:
       - Reflects the search intent and semantic meaning.
       - Groups related keywords cohesively and logically.
       - Incorporates relevant keywords naturally (no forced usage).
       - Avoids redundant or repetitive terms.
       - Expands into a natural sentence when using multiple keywords.
       - Flows logically from the previous heading.
    2. Include 1 or 2 additional semantically relevant content subsections that enrich the topic while maintaining alignment with the article's intent.
    3. Provide the English translation of the H2 for Perplexity search.
    4. List target keywords with their search volumes that belong in this section.
    5. Combine reasoning, context, and guidelines into a single 'Details' field explaining the section's purpose and how to handle it.
    6. List 3-5 related concepts to enhance semantic coverage.
    7. Suggest the best content format (e.g., paragraphs, lists, tables, step-by-step guides).

    For the **FAQs section**:
    1. Create an H2 heading in Thai: "คำถามที่พบบ่อยเกี่ยวกับ [main keyword]"
    2. Provide the English translation: "Frequently Asked Questions About [main keyword]"
    3. List 5-7 semantically relevant questions in Thai that:
       - Address common queries related to the main topic.
       - Clarify complex concepts in simple terms.
       - Avoid duplicating information already covered in other sections.
       - **Use modern Thai phrasing and avoid outdated pronouns such as "ฉัน".**
    4. Do NOT provide answers to the questions. Leave the answers blank for the content editor to research and fill in.
    5. For each question, generate a Perplexity research link using the English translation of the question, followed by "briefly explain in Thai".

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
        {{
            "heading": "คำถามที่พบบ่อยเกี่ยวกับ [main keyword]",
            "headingEnglish": "Frequently Asked Questions About [main keyword]",
            "targetKeywords": [],
            "relatedConcepts": [],
            "contentFormat": "Q&A format",
            "details": [
                {{
                    "question": "Question 1 in Thai (modern phrasing, e.g. หาข้อมูลเกี่ยวกับ [main keyword] ใหม่ๆ ได้ที่ไหน?)",
                    "researchLink": "https://www.perplexity.ai/search?q=English+translation+of+question+1+briefly+explain+in+Thai"
                }},
                {{
                    "question": "Question 2 in Thai",
                    "researchLink": "https://www.perplexity.ai/search?q=English+translation+of+question+2+briefly+explain+in+Thai"
                }}
                // More questions...
            ],
            "perplexityLink": "https://www.perplexity.ai/search?q=Frequently+Asked+Questions+About+[main keyword]+.+Explain+in+Thai"
        }}
        // More sections if needed...
    ]
    """
    structure_response = make_gemini_request(content_structure_prompt)
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

            if promotional_entities:
                query = f"{section['headingEnglish']} {' '.join(promotional_entities)}. Explain in Thai"
            else:
                query = f"{section['headingEnglish']}. Explain in Thai"
            section["perplexityLink"] = "https://www.perplexity.ai/search?q=" + quote_plus(query)

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse content structure response as JSON. Response: {structure_response[:200]}...") from e
    except ValueError as e:
        raise Exception(f"Invalid content structure format: {str(e)}") from e
    except Exception as e:
        raise Exception(f"Error generating content structure: {str(e)}") from e

    seo_plan = SEOPlan(
        title_meta_h1=TitleMetaH1(**title_meta_h1),
        content_structure=[Section(**section) for section in content_structure]
    )

    return seo_plan

# Streamlit UI
st.set_page_config(page_title="SEO Content Recommender", layout="wide")
st.title("SEO Content Recommender")

brief_type = st.radio(
    "Select Brief Type:",
    options=["Internal", "External"],
    index=0,
    key="brief_type"
)

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

                intent_output = generate_intent(keywords)

                st.session_state.keywords = keywords
                st.session_state.content_intents = intent_output.intents
                st.session_state.intent_generated = True

                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter some keywords first.")

if st.session_state.intent_generated:
    st.subheader("Select Content Intent:")
    st.write("Choose one of the following content intents that best matches your goals:")
    
    selected_intent = st.radio(
        "Available Intents:",
        options=st.session_state.content_intents,
        index=0,
        format_func=lambda x: f"{x}",
        key="intent_selector"
    )
    
    edited_intent = st.text_area(
        "Review and edit the selected content intent if needed:",
        value=selected_intent,
        height=100,
        key="intent_editor"
    )
    st.session_state.content_intent = edited_intent

if st.session_state.intent_generated:
    if st.button("Analyze Keyword Relevancy"):
        try:
            with st.spinner("Analyzing keyword relevancy..."):
                if not st.session_state.analyzed_keywords:
                    analyzed = analyze_keyword_relevancy(st.session_state.keywords)
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

                st.markdown("### Selected Keywords")
                for kw in selected_keywords:
                    st.markdown(f"- **{kw['keyword']}**: {kw['volume']:,}")

                total_volume = sum(kw['volume'] for kw in selected_keywords)
                st.markdown(f"**Total Monthly Search Volume**: {total_volume:,}")
                st.markdown("---")

                keyword_input_str = "\n".join([
                    f"{kw['keyword']}|{kw['volume']}"
                    for kw in selected_keywords
                ])

                seo_plan = full_seo_plan(KeywordInput(raw_input=keyword_input_str))

                st.markdown("## SEO Content Plan")
                st.markdown(f"**Title**: {seo_plan.title_meta_h1.title}")
                st.markdown(f"**Meta Description**: {seo_plan.title_meta_h1.metaDescription}")
                st.markdown(f"**H1**: {seo_plan.title_meta_h1.h1}")
                st.markdown("## Content Structure")
                for section in seo_plan.content_structure:
                    heading = re.sub(r'\s*\(\d+\)\s*', '', section.heading)
                    st.markdown(f"## **{heading}**")
                    
                    target_keywords = "<span style='color: black;'>, </span>  ".join(section.targetKeywords)
                    st.markdown(
                        f"""
                        <span style='color: black;'>Target Keywords:</span> 
                        <span style='color: green;'>{target_keywords}</span>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("**Related Concepts**: " + ", ".join(section.relatedConcepts))
                    st.markdown(f"**Content Format**: {section.contentFormat}")
                    
                    if isinstance(section.details, list):  # FAQ section
                        st.markdown("**Guidelines:**")
                        for qa in section.details:
                            st.markdown(f"- **{qa['question']}**")
                            if st.session_state.brief_type == "Internal":
                                st.markdown(f"  Research: [{qa['researchLink']}]({qa['researchLink']})")
                    else:
                        st.markdown(f"**Guidelines:** {section.details}")
                        if st.session_state.brief_type == "Internal":
                            st.markdown(f"**Research:** [{section.perplexityLink}]({section.perplexityLink})")
                    
                    st.markdown("---")

                if st.session_state.brief_type == "External":
                    external_guidelines = """
กฎหลักๆ:
- เนื้อหาความยาว ประมาณ 2000 คำ
- ในแต่ละหัวข้อ ใช้แต่ละคีย์ แค่ 1-2 ครั้ง
- หลีกเลี่ยงการคัดลอกเนื้อหาจาก AI
- หาภาพประกอบความละเอียดสูง (>1200px) 
- ใช้อ้างอิงจากเว็บไซต์น่าเชื่อถือด้วย in-text citation ไม่เน้นการอ้างอิงท้ายบทความ

วิธีที่ถูกต้อง:
USDT หรือ Tether เป็นสกุลเงินดิจิทัลประเภท stablecoin ที่มีมูลค่าผูกติดกับดอลลาร์สหรัฐฯ โดยมีจุดประสงค์เพื่อรักษาเสถียรภาพของมูลค่าให้เท่ากับ 1 ดอลลาร์สหรัฐฯ ต่อ 1 USDT (Investopedia)

วิธีที่ไม่ถูกต้อง:
USDT หรือ Tether เป็นสกุลเงินดิจิทัลประเภท stablecoin ที่มีมูลค่าผูกติดกับดอลลาร์สหรัฐฯ โดยมีจุดประสงค์เพื่อรักษาเสถียรภาพของมูลค่าให้เท่ากับ 1 ดอลลาร์สหรัฐฯ ต่อ 1 USDT

……….

อ้างอิง:
- https://www.investopedia.com/
- https://www.techopedia.com/th
"""
                    st.markdown("## Additional Guidelines for External Brief")
                    st.markdown(external_guidelines)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
