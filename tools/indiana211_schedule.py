from __future__ import annotations

import json
import re
from dataclasses import dataclass


DAY_VALUES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_ALIASES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

TIME_PATTERN = r"(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)"
SCHEDULE_RANGE_RE = re.compile(
    rf"(?P<days>(?:mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|r|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|daily|every day|weekdays|weekends|[\s,;&-])+?)\s+"
    rf"(?P<start>{TIME_PATTERN})\s*-\s*(?P<end>{TIME_PATTERN})",
    re.IGNORECASE,
)
DAY_RANGE_RE = re.compile(
    r"\b(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|r|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\s*-\s*"
    r"(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|r|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScheduleWindow:
    day: str
    start_minute: int
    end_minute: int


def schedule_status(text: object) -> str:
    text = _clean(text).lower()
    if schedule_windows(text):
        return "structured"
    if "by appointment" in text or "appointment only" in text or "by appoin" in text:
        return "appointment_only"
    if text:
        return "unsupported"
    return "unknown"


def schedule_windows(text: object) -> tuple[ScheduleWindow, ...]:
    text = _clean(text).lower().replace("–", "-").replace("—", "-")
    windows: list[ScheduleWindow] = []
    if re.search(r"\bdaily\s+24\s*hours?\b|\b24/7\b|\b24 hours\b", text):
        windows.extend(ScheduleWindow(day, 0, 24 * 60) for day in DAY_VALUES)
    for match in SCHEDULE_RANGE_RE.finditer(text):
        days = _parse_days(match.group("days"))
        start = parse_schedule_time(match.group("start"))
        end = parse_schedule_time(match.group("end"))
        if not days or start is None or end is None:
            continue
        windows.extend(ScheduleWindow(DAY_VALUES[day], start, end) for day in sorted(days))
    return tuple(_dedupe_windows(windows))


def schedule_windows_from_json(row: dict) -> tuple[ScheduleWindow, ...]:
    raw_windows = row.get("schedule_windows")
    if isinstance(raw_windows, str) and raw_windows.strip():
        try:
            raw_windows = json.loads(raw_windows)
        except ValueError:
            raw_windows = None
    if isinstance(raw_windows, list):
        windows = []
        for item in raw_windows:
            if not isinstance(item, dict):
                continue
            day = str(item.get("day", "")).lower()
            if day not in DAY_VALUES:
                continue
            start = _window_minute(item, "start")
            end = _window_minute(item, "end")
            if start is None or end is None:
                continue
            windows.append(ScheduleWindow(day, start, end))
        return tuple(windows)
    return schedule_windows(row.get("site_schedule", ""))


def _window_minute(item: dict, key: str) -> int | None:
    minute_key = f"{key}_minute"
    if minute_key in item:
        try:
            return int(item.get(minute_key))
        except (TypeError, ValueError):
            return None
    value = str(item.get(key, ""))
    if key == "end" and value == "24:00":
        return 24 * 60
    return parse_24_hour_time(value)


def parse_24_hour_time(value: str) -> int | None:
    text = _clean(value)
    if not text:
        return None
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def parse_schedule_time(value: str) -> int | None:
    text = _clean(value).lower().replace(" ", "")
    if text == "noon":
        return 12 * 60
    if text == "midnight":
        return 24 * 60
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem == "am" and hour == 12:
        hour = 0
    if meridiem == "pm" and hour != 12:
        hour += 12
    return hour * 60 + minute


def format_minutes(value: int) -> str:
    hour = value // 60
    minute = value % 60
    return f"{hour:02d}:{minute:02d}"


def is_24_hour_window(window: ScheduleWindow) -> bool:
    return window.start_minute == 0 and window.end_minute >= 24 * 60


def _parse_days(value: str) -> set[int]:
    text = value.lower().replace(" and ", ",")
    if "daily" in text or "every day" in text:
        return set(range(7))
    if "weekdays" in text:
        return set(range(5))
    if "weekends" in text:
        return {5, 6}
    days = set()
    for start, end in DAY_RANGE_RE.findall(text):
        start_idx = DAY_ALIASES[_norm(start)]
        end_idx = DAY_ALIASES[_norm(end)]
        if start_idx <= end_idx:
            days.update(range(start_idx, end_idx + 1))
        else:
            days.update(range(start_idx, 7))
            days.update(range(0, end_idx + 1))
    for token in re.split(r"[,\s;&-]+", text):
        normalized = _norm(token)
        if normalized in DAY_ALIASES:
            days.add(DAY_ALIASES[normalized])
    return days


def _dedupe_windows(windows: list[ScheduleWindow]) -> list[ScheduleWindow]:
    deduped = []
    seen = set()
    for window in windows:
        key = (window.day, window.start_minute, window.end_minute)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)
    return deduped


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _norm(value: str) -> str:
    return value.lower().strip().replace(".", "").replace("-", "_").replace(" ", "_")
