from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from aiogram.types import Message

from timeline_hub.handlers.clips.common import extract_clip_file_id
from timeline_hub.handlers.clips.flow import store_allowed_seasons, year_option_universe
from timeline_hub.services.clip_store import ClipGroup, Season, SubSeason, Universe
from timeline_hub.services.message_buffer import MessageGroup
from timeline_hub.settings import Settings


@dataclass(slots=True)
class RouteBatch:
    clip_group: ClipGroup
    sub_season: SubSeason
    messages: list[Message]


def plan_route_batches(
    message_groups: Sequence[MessageGroup],
    *,
    settings: Settings,
) -> tuple[list[RouteBatch], str | None]:
    batches: list[RouteBatch] = []
    current_route: tuple[ClipGroup, SubSeason] | None = None
    today = date.today()
    allowed_years = set(
        year_option_universe(
            current_year=today.year,
            min_year=settings.min_year,
        )
    )

    for message_group in message_groups:
        for message in message_group:
            clip_file_id = extract_clip_file_id(message)
            if clip_file_id is None:
                if message.text is None:
                    continue

                parsed_route = _parse_and_validate_route(
                    message.text,
                    today=today,
                    allowed_years=allowed_years,
                )
                if parsed_route is None:
                    continue
                current_route = parsed_route
                continue

            route_text = message.caption if message.caption is not None else message.text
            if route_text is None:
                if current_route is None:
                    return [], 'Missing route text'
                next_route = current_route
            else:
                next_route = _parse_and_validate_route(
                    route_text,
                    today=today,
                    allowed_years=allowed_years,
                )
                if next_route is None:
                    return [], 'Invalid route text'

            next_group, next_sub_season = next_route
            if not batches or batches[-1].clip_group != next_group or batches[-1].sub_season != next_sub_season:
                batches.append(
                    RouteBatch(
                        clip_group=next_group,
                        sub_season=next_sub_season,
                        messages=[message],
                    )
                )
            else:
                batches[-1].messages.append(message)
            current_route = next_route

    return batches, None


def parse_route_text(text: str) -> tuple[ClipGroup, SubSeason] | None:
    return parse_group_selector_text(text, allow_sub_season_suffix=True)


def parse_group_selector_text(
    text: str,
    *,
    allow_sub_season_suffix: bool,
) -> tuple[ClipGroup, SubSeason] | None:
    normalized = text.strip()
    if len(normalized) < 3:
        return None

    universe = Universe.WEST
    route_text = normalized
    if normalized[0].lower() in {'w', 'e'}:
        universe = Universe.WEST if normalized[0].lower() == 'w' else Universe.EAST
        route_text = normalized[1:]

    expected_lengths = {3, 4} if allow_sub_season_suffix else {3}
    if len(route_text) not in expected_lengths:
        return None

    year_suffix = route_text[0:2]
    season_text = route_text[2]
    suffix_text = route_text[3:] if len(route_text) == 4 else ''

    if not year_suffix.isdigit() or not season_text.isdigit():
        return None

    try:
        season = Season(int(season_text))
    except ValueError:
        return None

    sub_season = SubSeason.NONE
    if suffix_text:
        if len(suffix_text) != 1:
            return None
        suffix = suffix_text.lower()
        if suffix not in {'a', 'b', 'c', 'd'}:
            return None
        sub_season = SubSeason(suffix.upper())

    return ClipGroup(universe=universe, year=2000 + int(year_suffix), season=season), sub_season


def _parse_and_validate_route(
    text: str,
    *,
    today: date,
    allowed_years: set[int],
) -> tuple[ClipGroup, SubSeason] | None:
    parsed_route = parse_route_text(text)
    if parsed_route is None:
        return None
    parsed_group, _sub_season = parsed_route
    if parsed_group.year not in allowed_years:
        return None
    if parsed_group.season not in store_allowed_seasons(year=parsed_group.year, today=today):
        return None
    return parsed_route
