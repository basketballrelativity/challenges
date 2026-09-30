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
from datetime import datetime, time
from enum import Enum
from typing import Optional

import pandas as pd

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain.agents import create_agent

SEASON = "2024-25"
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
- If available, you'll also be provided a JPG image of the start of the video that you can use to help determine the challenging team.
Try not to rely on this unless the challenging team isn't indicated in the website text. If this image isn't available, it won't be passed
along to you
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
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": challenge["thumbnail_url"]}}
                        ],
                    }
                ]
            }
        )

        return result["structured_response"]

    except Exception as exc:
        try:
            print(f"Could not extract {challenge["permalink"]}: {exc}")
            print("Trying without thumbnail image")
            result = extractor.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ],
                        }
                    ]
                }
            )

            return result["structured_response"]
        except Exception as exc:
            print(f"Failed again for {challenge["permalink"]}: {exc}")
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


def challenge_agent_eval():
    """ This function runs evaluation on a subset
    of known challenge data to test the performance
    of the agent
    """
    # Eval dates
    start_date = "10/25/2023"
    end_date = "10/27/2023"

    # Filter known data
    challenge_df = pd.read_csv("challenge_23_24.csv")
    challenge_df = challenge_df[
        (challenge_df["Date"]>=start_date.replace("202", "2")) &
        (challenge_df["Date"]<=end_date.replace("202", "2"))
    ]

    # Agent
    challenges = extract_challenges(
        start_date,
        end_date
    )
    agent_df = pd.DataFrame([vars(x) for x in challenges])
    # Process classes
    for col in ["visiting", "home", "initial_call", "challenging_team", "outcome"]:
        agent_df[col] = [x.value for x in agent_df[col]]

    # Rename agent_df columns
    rename_dict = {
        "date": "Date",
        "visiting": "Visiting",
        "home": "Home",
        "challenging_team": "Challenging Team",
        "period": "Period",
        "time": "Time"
    }
    agent_df = agent_df.rename(columns=rename_dict)
    agent_df["Date"] = [x.replace("202", "2") for x in agent_df["Date"]]

    # Process time
    challenge_df["processed_time"] = [
        time(
            hour=0,
            minute=int(x.split(":")[0]),
            second=int(float(x.split(":")[-1])),
        ) for x in challenge_df["Time"]
    ]
    agent_df["processed_time"] = [
        time(
            hour=0,
            minute=int(x.split(":")[0]),
            second=int(float(x.split(":")[-1])),
        ) for x in agent_df["Time"]
    ]
    # Process calls
    challenge_df["processed_calls"] = [
        "Foul" if "Foul" in x else
        "Possession" if "Possession" in x else
        "Goaltending" for x in challenge_df["Initial Call"]
    ]

    comb_df = agent_df.merge(
        challenge_df,
        on=["Date", "Visiting", "Home", "Challenging Team",
        "Period", "processed_time"
        ],
        how="left"
    )

    # Time to evaluate!
    matches = len(comb_df[pd.notnull(comb_df["processed_calls"])])
    possible_matches = len(challenge_df)

    # Number of matches and correct call type
    print(f"Matches: {matches}/{possible_matches}")
    correct_calls = sum(comb_df["processed_calls"] == comb_df["initial_call"])
    print(f"Correct Call Type: {correct_calls}/{possible_matches}")

    return comb_df, agent_df, challenge_df


if __name__ == "__main__":

    challenges = extract_challenges(
        "04/15/2025",
        "06/22/2025",
    )

    agent_df = pd.DataFrame([vars(x) for x in challenges])
    # Process classes
    for col in ["visiting", "home", "initial_call", "challenging_team", "outcome"]:
        agent_df[col] = [x.value for x in agent_df[col]]

    rename_dict = {
        "date": "Date",
        "visiting": "Visiting",
        "home": "Home",
        "challenging_team": "Challenging Team",
        "period": "Period",
        "time": "Time",
        "initial_call": "Initial Call",
        "link": "Link",
        "outcome": "Outcome"
    }

    agent_df = agent_df.rename(columns=rename_dict)
    agent_df["Date"] = [x.replace("202", "2") for x in agent_df["Date"]]

    agent_df.to_csv(f"challenges/months/challenge_playoffs_{SEASON}.csv", index=False)