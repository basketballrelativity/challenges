"""
Extract NBA Coach's Challenges from the NBA Official replay archive.

Architecture:
    NBA archive
        -> discover challenge pages
        -> fetch individual pages
        -> LangChain structured extraction
        -> Challenge objects
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain.agents import create_agent

SEASON = "2025-26"
NBA_ARCHIVE = "https://official.nba.com/wp-json/api/v1/query_replay_feed"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://official.nba.com/replay/archive/",
}


# ---------------------------------------------------------------------------
# Your enums
# ---------------------------------------------------------------------------

class TeamEnum(str, Enum):
    pacers = "IND"
    pistons = "DET"
    warriors = "GSW"
    clippers = "LAC"
    lakers = "LAL"
    rockets = "HOU"
    nets = "BKN"
    trailblazers = "POR"
    timberwolves = "MIN"
    celtics = "BOS"
    heat = "MIA"
    knicks = "NYK"
    bucks = "MIL"
    kings = "SAC"
    suns = "PHX"
    nuggets = "DEN"
    mavericks = "DAL"
    wizards = "WAS"
    grizzlies = "MEM"
    bulls = "CHI"
    spurs = "SAS"
    sixers = "PHI"
    jazz = "UTA"
    pelicans = "NOP"
    hornets = "CHA"
    hawks = "ATL"
    thunder = "OKC"
    magic = "ORL"
    cavaliers = "CLE"
    raptors = "TOR"


class CallEnum(str, Enum):
    foul = "Foul"
    possession = "Possession"
    goaltending = "Goaltending"


class OutcomeEnum(str, Enum):
    won = "Won"
    lost = "Lost"


class Challenge(BaseModel):
    date: str = Field(
        description="Date of the game in MM/DD/YYYY format"
    )

    visiting: TeamEnum = Field(
        description="Three-letter NBA code for visiting team"
    )

    home: TeamEnum = Field(
        description="Three-letter NBA code for home team"
    )

    initial_call: CallEnum = Field(
        description="Type of call being challenged"
    )

    challenging_team: TeamEnum = Field(
        description="Team that initiated the Coach's Challenge"
    )

    outcome: OutcomeEnum = Field(
        description="Won if the challenge resulted in the original ruling "
                    "being overturned; Lost otherwise"
    )

    period: int = Field(
        description="Quarter/period. OT1 is 5, OT2 is 6, etc."
    )

    time: str = Field(
        description="Time remaining in period, e.g. 09:22.0"
    )

    link: str = Field(
        description="URL of the NBA replay page"
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

session = requests.Session()
session.headers.update(HEADERS)


def get_page(url: str) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


# ---------------------------------------------------------------------------
# Archive discovery
# ---------------------------------------------------------------------------

def discover_links(url: str) -> list[str]:
    """
    Return all links on an NBA Official page.
    """
    soup = get_page(url)

    links = []

    for anchor in soup.select("permalink"):
        href = anchor.get("permalink")

        if not href:
            continue

        if href.startswith("/"):
            href = "https://official.nba.com" + href

        if href.startswith("https://official.nba.com"):
            links.append(href)

    return sorted(set(links))


def discover_challenge_pages(start_date: str, end_date: str) -> list[str]:
    """
    Discover individual NBA Coach's Challenge replay pages.

    Dates are MM/DD/YYYY.
    """

    params = {
        "offset": 0,
        "date_range[from]": start_date,
        "date_range[to]": end_date,
        "team": "",
        "season": SEASON,
        "trigger": "coachs-challenge",
        "outcome": 0,
    }

    # First find the season/archive pages.
    response = requests.get(NBA_ARCHIVE, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    archive_links = response.json()

    return archive_links


# ---------------------------------------------------------------------------
# Extract page text
# ---------------------------------------------------------------------------

def page_to_text(url: str) -> str:
    """
    Convert an NBA replay page to clean text for the LLM.
    """

    soup = get_page(url)

    # Remove things that aren't useful to the model.
    for element in soup(
        ["script", "style", "noscript", "svg"]
    ):
        element.decompose()

    text = soup.get_text("\n")

    # Normalize whitespace.
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
    ]

    lines = [line for line in lines if line]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You extract NBA Coach's Challenge information from NBA Official replay pages.

Return information for just one challenge

Important:

- Do not invent information.
- Use only information contained in the supplied NBA page and returned links.
- Convert team names to their three-letter NBA codes.
- Convert the game date to MM/DD/YYYY.
- Convert the time to MM:SS.T format (T is tenths of a second remaining)
-- You'll see this in the website page text at the beginning of the text describing the play
- Convert the period to an integer:
    1 = first quarter
    2 = second quarter
    3 = third quarter
    4 = fourth quarter
    5 = OT1
    6 = OT2
    etc.
-- You'll see this right after the game time above and before the type of challenge described
- Convert the challenge type into one of the available fields (Foul, Possession, or Goaltending)
-- Goaltending is also referred to as basket interference
- Convert challenge results into:
    Won = the challenge successfully changed the ruling (you'll likely see "overturned" in the text description)
    Lost = the original ruling was upheld (you'll see references to the call standing or being confirmed)
- The link must be the supplied NBA URL.
"""


def build_extractor():
    """
    Create a LangChain agent whose job is structured extraction,
    not web browsing.
    """

    return create_agent(
        model="openai:gpt-5.5",
        response_format=Challenge,
        system_prompt=SYSTEM_PROMPT,
    )


def extract_challenge(
    extractor,
    challenge
) -> Optional[Challenge]:

    text = page_to_text(challenge["permalink"])
    prompt = f"""
Extract the Coach's Challenge information at this URL

URL:
{challenge["permalink"]}

Link text:
{text}
"""

    try:
        result = extractor.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )

        return result["structured_response"]

    except Exception as exc:
        print(f"Could not extract {challenge["permalink"]}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_challenges(
    start_date: str,
    end_date: str,
) -> list[Challenge]:

    print(
        f"Searching for challenges from "
        f"{start_date} through {end_date}"
    )

    challenges = discover_challenge_pages(start_date, end_date)

    extractor = build_extractor()

    confirmed_challenges = []

    for challenge in challenges:
        challenge = extract_challenge(extractor, challenge)

        if challenge is None:
            continue

        confirmed_challenges.append(challenge)

    return confirmed_challenges


if __name__ == "__main__":

    challenges = extract_challenges(
        "06/13/2026",
        "06/13/2026",
    )

    for challenge in challenges:
        print(challenge.model_dump_json(indent=2))