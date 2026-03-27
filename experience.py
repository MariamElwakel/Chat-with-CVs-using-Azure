import re
from datetime import date
from dateutil.relativedelta import relativedelta # to calculate difference between two dates --> more accurate than subtract

# Date Parsing
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}

# Date Pattern
DATE_TOKEN = (
    r"(?:"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\.?\s+\d{4}"           # Month YYYY
    r"|\d{1,2}[/\-]\d{4}"   # MM/YYYY
    r"|\d{4}[/\-]\d{1,2}"   # YYYY/MM
    r"|\d{4}"                # YYYY alone
    r"|present|current|now|today"
    r")"
)

# Keywords that signal a work experience section header
EXPERIENCE_KEYWORDS = {
    "experience", "employment", "work history", "career",
    "professional background", "positions", "jobs"
}

# Experience patterns in the user query
EXP_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*\+\s*years?",
    r"(?:more than|over|greater than)\s+(\d+(?:\.\d+)?)\s*years?",
    r"(?:at least|minimum|min)\s+(\d+(?:\.\d+)?)\s*years?",
    r"(\d+(?:\.\d+)?)\s*years?\s+(?:of\s+)?experience",
]


def parse_date(date_token: str) -> date | None:
    """
    Parse a single date extracted from a CV into a date object.
    Handles: 'present'/'current'/'now', 'YYYY', 'MM/YYYY', 'Month YYYY', 'YYYY-MM'
    """
    date_token = date_token.strip().lower()

    if date_token in ("present", "current", "now", "today", "–", "-", ""):
        return date.today()

    # YYYY only → Assumes January 1st of that year
    if re.fullmatch(r"\d{4}", date_token):
        return date(int(date_token), 1, 1)

    # MM/YYYY or MM-YYYY
    date_match = re.fullmatch(r"(\d{1,2})[/\-](\d{4})", date_token)
    if date_match:
        return date(int(date_match.group(2)), int(date_match.group(1)), 1)

    # YYYY/MM or YYYY-MM
    date_match = re.fullmatch(r"(\d{4})[/\-](\d{1,2})", date_token)
    if date_match:
        return date(int(date_match.group(1)), int(date_match.group(2)), 1)

    # Month YYYY (April 2019)
    date_match = re.fullmatch(r"([a-z]+)\.?\s+(\d{4})", date_token)
    if date_match and date_match.group(1).rstrip(".") in MONTHS:
        return date(int(date_match.group(2)), MONTHS[date_match.group(1).rstrip(".")], 1)

    # YYYY Month (2019 April)
    date_match = re.fullmatch(r"(\d{4})\s+([a-z]+)\.?", date_token)
    if date_match and date_match.group(2).rstrip(".") in MONTHS:
        return date(int(date_match.group(1)), MONTHS[date_match.group(2).rstrip(".")], 1)

    return None


def is_experience_chunk(doc) -> bool:
    """
    Check whether a document chunk belongs to a work experience section.
    Return True if any header (1–3) contains experience-related keywords,
    otherwise False.
    """
    meta = doc.metadata
    for key in ("Header 1", "Header 2", "Header 3"):
        header_val = meta.get(key, "").lower()
        if any(kw in header_val for kw in EXPERIENCE_KEYWORDS):
            return True
    return False


def calculate_years_of_experience(md_docs) -> float | None:
    """
    Extract date ranges ONLY from chunks whose Header 1/2/3 indicates
    a work experience section, then merge overlapping intervals to avoid
    double-counting concurrent roles, and return total years as a float.
    Returns None if no valid date ranges are found in experience sections.
    """
    # Join only the experience-section chunks into one text block
    experience_text = "\n\n".join(
        doc.page_content
        for doc in md_docs
        if is_experience_chunk(doc)
    )

    # If no experience section, return None
    if not experience_text.strip():
        return None

    # Compile regex to match date ranges
    DATE_RANGE_PATTERN = re.compile(
        rf"({DATE_TOKEN})\s*(?:–|—|-{{1,2}}|to)\s*({DATE_TOKEN})",
        re.IGNORECASE # for case-insensitive
    )

    # Extract valid (start, end) date intervals from text
    intervals = []
    for date_match in DATE_RANGE_PATTERN.finditer(experience_text):
        
        start_date = parse_date(date_match.group(1))
        end_date = parse_date(date_match.group(2))

        if start_date and end_date and start_date <= end_date:
            intervals.append((start_date, end_date))

    # If no valid intervals found, return None
    if not intervals:
        return None

    # Sort and merge overlapping intervals to avoid double-counting concurrent roles
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]

    for start_date, end_date in intervals[1:]:

        # If current job starts before previous ends → Extend the end date
        if start_date <= merged[-1][1]:                          
            merged[-1][1] = max(merged[-1][1], end_date)
        else:
            merged.append([start_date, end_date])

    # Sum durations across all merged intervals
    total_months = sum(
        relativedelta(end_date, start_date).months + relativedelta(end_date, start_date).years * 12
        for start_date, end_date in merged
    )

    return round(total_months / 12, 1) if total_months > 0 else None


def get_experience_chunk(candidate_name: str, chunks: list) -> str | None:
    """
    Find the experience section chunk for a specific candidate
    by matching candidate_name in metadata and header keywords.
    """

    for chunk in chunks:
        if chunk.metadata.get("candidate_name") != candidate_name:
            continue

        # Check if any header in this chunk's metadata is an experience header
        for key in ("Header 1", "Header 2", "Header 3"):
            header_val = chunk.metadata.get(key, "").lower()
            if any(kw in header_val for kw in EXPERIENCE_KEYWORDS):
                return chunk.page_content
            
    return None


def extract_min_years(query: str) -> float | None:
    """
    Parse experience thresholds from recruiter queries.
    Handles:
      - '5+ years'
      - 'more than 3 years' / 'over 3 years' / 'greater than 3 years'
      - 'at least 7 years' / 'minimum 4 years' / 'min 4 years'
      - '3 years of experience' / '3 years experience'
    Returns the numeric threshold as a float.
    """
    query_lower = query.lower()
    
    for pattern in EXP_PATTERNS:
        exp_match = re.search(pattern, query_lower)
        if exp_match:
            return float(exp_match.group(1))
        
    return None