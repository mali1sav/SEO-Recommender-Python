import os
import re
import json
from urllib.parse import quote_plus  # สำหรับการเข้ารหัส URL

import requests
from requests.adapters import Retry, HTTPAdapter
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
load_dotenv()

# Initialize OpenRouter client
def init_openrouter_client():
    """Initialize OpenRouter configuration."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your environment variables.")
        return None

    return {
        'api_key': api_key,
        'api_url': os.getenv('OPENROUTER_API_URL', 'https://openrouter.ai/api/v1/chat/completions'),
        'model': os.getenv('OPENROUTER_MODEL', 'google/gemini-2.5-flash-preview-09-2025'),
        'site_url': os.getenv('OPENROUTER_SITE_URL', ''),
        'app_name': os.getenv('OPENROUTER_APP_NAME', 'SEO Content Recommender')
    }

# Initialize OpenRouter configuration
openrouter_client = init_openrouter_client()
if not openrouter_client:
    raise EnvironmentError("Failed to initialize OpenRouter client.")

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
    Returns provided promotional entities, or defaults to ["Best Wallet"].
    Emphasizes user-provided entities without any brand rotation.
    """
    custom_entities_raw = st.session_state.get("custom_promotional_entities", "").strip()
    if custom_entities_raw:
        return [entity.strip() for entity in re.split(r'[\n,]+', custom_entities_raw) if entity.strip()]

    # Default emphasis entity when none provided
    return ["Best Wallet"]

session = create_session()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=20),
    retry=tenacity.retry_if_exception_type((Exception)),
    retry_error_callback=lambda retry_state: None
)
def make_openrouter_request(prompt: str) -> str:
    """Make OpenRouter API request with retries and proper error handling."""
    try:
        headers = {
            "Authorization": f"Bearer {openrouter_client['api_key']}",
            "Content-Type": "application/json"
        }
        if openrouter_client.get('site_url'):
            headers["HTTP-Referer"] = openrouter_client['site_url']
        if openrouter_client.get('app_name'):
            headers["X-Title"] = openrouter_client['app_name']

        payload = {
            "model": openrouter_client['model'],
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = session.post(
            openrouter_client['api_url'],
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 429:
            detail = response.json().get('error', {}).get('message', response.text)
            raise RuntimeError(f"OpenRouter rate limit reached: {detail}")

        response.raise_for_status()
        data = response.json()

        choices = data.get('choices')
        if not choices:
            raise ValueError("OpenRouter response did not include choices.")

        message_content = choices[0].get('message', {}).get('content')
        if not message_content:
            raise ValueError("OpenRouter response missing message content.")

        if isinstance(message_content, list):
            text = "".join(part.get('text', '') for part in message_content)
        else:
            text = message_content

        return text.strip()
    except Exception as e:
        st.error(f"Error making OpenRouter request: {str(e)}")
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

# Brand rotation removed; emphasis on provided entity with default "Best Wallet".

def rotate_keywords(keywords: List[Keyword], num_groups=5) -> List[List[str]]:
    """Split keywords into rotating groups"""
    return [keywords[i::num_groups] for i in range(num_groups)]

def generate_intent(keywords: List[Keyword]) -> IntentOutput:
    """
    Generates content intent with enforced content types.
    Modified to generate longer, more detailed intents (around 2-3 sentences)
    to guide the content structure more comprehensively.
    """
    if not keywords:
        raise Exception("Keyword list is empty.")
        
    primary_keyword = keywords[0].term
    supporting_keywords = [kw.term for kw in keywords[1:]]
    keywords_list = "\n".join([kw.term for kw in keywords])
    supporting_text = ", ".join(supporting_keywords) if supporting_keywords else "(no supporting keywords provided)"

    prompt = f"""
    You are an SEO strategist who must always prioritise the primary keyword.
    Primary keyword (treat as the main topic and mention first in every intent without quotes/backticks): {primary_keyword}
    Supporting keywords (use naturally after the primary keyword): {supporting_text}
    Promotional entities (weave in only if helpful and natural; prefer earlier sections and concrete steps): {', '.join(extract_promotional_entities('')) or 'Best Wallet'}

    Generate 5 distinct Thai content intents that are comprehensive and detailed (around 2-3 sentences each). You can use the following example formats or come up with better ones:

    1. Listicle: "10 ข้อควรรู้เกี่ยวกับ [main keyword]" 
    2. How-To: "วิธี [action] [main keyword] แบบมืออาชีพ"
    3. Comparison: "[option1] vs [option2] ต่างกันอย่างไร?"
    4. Expert: "ผู้เชี่ยวชาญเผย [insight] เกี่ยวกับ [main keyword]"
    5. Problem-Solving: "แก้ไขปัญหา [common issue] ด้วย [solution]"

    Keywords in order of priority: {keywords_list}

    Return EXACTLY 5 intents in Thai that:
    1. Start by naturally mentioning the primary keyword within the first clause (first 12 words)
    2. Keep the primary keyword in plain text (no quotes, backticks, or brackets)
    3. Capture the main topic from a unique and engaging angle
    4. Address a variety of user questions and search intents
    5. Provide a detailed flow for content structure with context and additional information
    6. Where suitable, naturally mention one promotional entity (e.g., Best Wallet) as part of a concrete action or example, not an ad; avoid forced mentions.
    7. Prefer placing such mention earlier when it fits (e.g., step-by-step or tools section), not only in the last section.
    8. Use natural Thai language while retaining supporting keywords in their original form, whether in Thai or English
    9. Are comprehensive and detailed (around 2-3 sentences) to help in generating a rich content structure

    Return EXACTLY 5 intents in Thai, one per line, numbered from 1-5. Do not include any extra explanation.

    Example format of the response:
    1. ข้อมูลเกี่ยวกับ [topic] ใน [context] พร้อมเจาะลึกคุณสมบัติและวิธีเลือกใช้งานที่เหมาะสม
    2. วิธีการ [action] สำหรับ [target] ที่ช่วยเพิ่มประสิทธิภาพและตอบโจทย์ความต้องการอย่างครบถ้วน
    3. เปรียบเทียบ [topic] ระหว่าง [option1], [option2], และ [option3] พร้อมวิเคราะห์ข้อดีข้อเสียอย่างละเอียด
    4. แนวทางการ [topic] ที่ช่วยให้เข้าใจในเชิงลึก พร้อมแนะนำเทคนิคและวิธีการปฏิบัติจริง
    5. ความสำคัญของ [topic] ในบริบทของ [impact] พร้อมให้ข้อมูลที่ครบถ้วนและแนวทางการปรับปรุงในอนาคต
    """
    intent_response = make_openrouter_request(prompt)
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
        result = make_openrouter_request(prompt)
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
    promotional_entities_text = ", ".join(promotional_entities)

    # Step 3: Generate Title, Meta, and H1 with focus on main keyword
    helpful_clause = "    5. Keeps the tone helpful and aligned with the search intent without unnecessary promotion.\n"
    title_promo_clause = ""
    if promotional_entities:
        title_promo_clause = (
            f"    6. When it benefits the reader, naturally reference promotional entities such as {promotional_entities_text} to illustrate solutions or cautions without sounding salesy or forced.\n"
        )

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
{helpful_clause}{title_promo_clause}

    Return in this exact JSON format:
    {{
        "title": "Your SEO title here",
        "metaDescription": "Your meta description here",
        "h1": "Your H1 here"
    }}

    Return ONLY the JSON, no other text.
    """
    title_response = make_openrouter_request(title_prompt)
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

    content_structure_promo_guideline = ""
    if promotional_entities:
        content_structure_promo_guideline = (
            f"    4. When it helps readers achieve the intent, weave promotional entities such as {promotional_entities_text} into the most relevant sections. Keep the guidance informative and practical, avoiding sales language.\n"
            "       - Prefer placing the first mention in section 2 or 3 if appropriate (e.g., getting started, step-by-step, or tool setup).\n"
            "       - Aim to mention each promotional entity at least once where it fits naturally.\n"
            "       - Explain why it matters for the reader (e.g., strengths, cautions, or use cases).\n"
        )
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
{content_structure_promo_guideline}{recent_insights_guideline}
    For each section (except FAQs):
    1. Create an H2 heading in Thai that:
       - Reflects the search intent and semantic meaning.
       - Groups related keywords cohesively and logically.
       - Incorporates relevant keywords naturally (no forced usage).
       - Avoids redundant or repetitive terms.
       - Expands into a natural sentence when using multiple keywords.
       - Flows logically from the previous heading.
    2. Include 1 or 2 additional semantically relevant content subsections that enrich the topic while maintaining alignment with the article's intent. If a promotional entity fits, blend it into actionable steps or examples.
    3. Provide the English translation of the H2, ensure comprehensive and cover all the main points which is key for getting the most out of Perplexity search.
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
    structure_response = make_openrouter_request(content_structure_prompt)
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

            # Build a clean research query; keep entities out of the core question
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
if 'custom_promotional_entities' not in st.session_state:
    # Active default so it's used, not just a placeholder
    st.session_state.custom_promotional_entities = "Best Wallet"

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

st.text_input(
    "Optional promotional entities (comma or newline separated):",
    value=st.session_state.custom_promotional_entities,
    key="custom_promotional_entities"
)

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

                st.rerun()
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
