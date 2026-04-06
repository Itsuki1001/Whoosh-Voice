import os
import time
import requests
import dateparser
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from ics import Calendar

from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

load_dotenv()

# ==================================================
# ICS CALENDAR
# ==================================================
EMBED_MODEL="text-embedding-3-small"
ICS_REFRESH_INTERVAL = 5 * 60

_calendar = None
_calendar_last_loaded = 0


def get_calendar() -> Calendar | None:
    global _calendar, _calendar_last_loaded
    now = time.time()
    if _calendar is None or (now - _calendar_last_loaded) > ICS_REFRESH_INTERVAL:
        ics_url = os.getenv("ics_url")
        try:
            response = requests.get(ics_url, timeout=5)
            response.raise_for_status()
            _calendar = Calendar(response.text)
            _calendar_last_loaded = now
            print("📅 Calendar refreshed")
        except Exception as e:
            print(f"⚠️ Calendar refresh failed: {e}")
    return _calendar


# ==================================================
# RAG (FAISS)
# ==================================================

def setup_rag():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.load_local(
        str(Path(__file__).parent.parent / "index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = setup_rag()


def _format_rag_docs(docs: list) -> str:
    grouped = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        grouped[source].append(doc.page_content)
    return "\n\n".join(
        f"{src}:\n" + "\n".join(chunks)
        for src, chunks in grouped.items()
    )


@tool
def rag_tool(query: str) -> str:
    """Retrieve information about the homestay."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        return _format_rag_docs(docs)
    except Exception as e:
        return f"RAG_ERROR::{e}"


# ==================================================
# DISTANCE & TRAVEL TIME
# ==================================================

def _get_coords(place: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    res = requests.get(
        url, params=params,
        headers={"User-Agent": "TherapyCenterBot/1.0"},
        timeout=5,
    ).json()
    if res:
        return float(res[0]["lat"]), float(res[0]["lon"])
    return None, None


def _get_distance_time(origin, dest, mode="driving"):
    url = (
        f"https://router.project-osrm.org/route/v1/"
        f"{mode}/{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
        f"?overview=false"
    )
    res = requests.get(url, timeout=5).json()
    route = res["routes"][0]
    return route["distance"] / 1000, route["duration"] / 3600


@tool("get_distance_to_homestay")
def get_distance_to_homestay(origin: str, mode: str = "driving") -> str:
    """Get distance and travel time to the homestay."""
    try:
        destination = "Nedumbassery"
        origin_coords = _get_coords(origin)
        dest_coords = _get_coords(destination)
        if not all(origin_coords) or not all(dest_coords):
            return "Could not determine location coordinates."
        dist, time_hrs = _get_distance_time(origin_coords, dest_coords, mode)
        return (
            f"From {origin} to our homestay: "
            f"{dist:.1f} km, about {time_hrs:.1f} hours by {mode}."
        )
    except Exception as e:
        return f"Distance error: {e}"


# ==================================================
# ROOM AVAILABILITY
# ==================================================

@tool
def get_room_availability(start_time, end_time) -> str:
    """Check if the homestay is available between given dates."""
    try:
        if isinstance(start_time, str):
            start_time = dateparser.parse(start_time)
        if isinstance(end_time, str):
            end_time = dateparser.parse(end_time)
        if not start_time or not end_time:
            return "Invalid date format."
        start_time = start_time.replace(tzinfo=timezone.utc)
        end_time = end_time.replace(tzinfo=timezone.utc)
        calendar = get_calendar()
        if not calendar:
            return "Availability system temporarily unavailable."
        for event in calendar.events:
            if not (end_time <= event.begin or start_time >= event.end):
                return "No"
        return "Yes"
    except Exception as e:
        return f"Availability error: {e}"


# ==================================================
# TOOL REGISTRY
# ==================================================

tools = [
    rag_tool,
    get_distance_to_homestay,
    get_room_availability,
]
