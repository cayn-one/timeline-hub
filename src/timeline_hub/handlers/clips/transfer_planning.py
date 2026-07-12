from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from aiogram.types import Message

from timeline_hub.handlers.clips.common import extract_clip_file_id
from timeline_hub.handlers.clips.flow import store_allowed_seasons, year_option_universe
from timeline_hub.handlers.clips.reconcile_input import parse_clip_identity_filename
from timeline_hub.handlers.clips.route_planning import parse_group_selector_text
from timeline_hub.services.clip_store import ClipGroup, ClipId
from timeline_hub.services.message_buffer import MessageGroup
from timeline_hub.settings import Settings


@dataclass(frozen=True, slots=True)
class TransferClipRef:
    source_group: ClipGroup
    clip_id: ClipId


@dataclass(slots=True)
class TransferBatch:
    destination_group: ClipGroup
    clips: list[TransferClipRef]


def plan_transfer_batches(
    message_groups: Sequence[MessageGroup],
    *,
    settings: Settings,
) -> tuple[list[TransferBatch], str | None]:
    batches: list[TransferBatch] = []
    current_destination: ClipGroup | None = None
    today = date.today()
    allowed_years = set(
        year_option_universe(
            current_year=today.year,
            min_year=settings.min_year,
        )
    )

    for message_group in message_groups:
        pending_destination = current_destination
        pending_clips: list[TransferClipRef] = []

        for message in message_group:
            clip_file_id = extract_clip_file_id(message)
            if clip_file_id is None:
                if message.text is None:
                    continue

                destination_text = message.text
                if _is_sub_group_route_text(destination_text):
                    return [], 'Physical group required'

                parsed_destination = _parse_and_validate_destination(
                    destination_text,
                    today=today,
                    allowed_years=allowed_years,
                )
                if parsed_destination is None:
                    continue
                if pending_clips:
                    _append_transfer_batch(
                        batches,
                        destination_group=pending_destination,
                        clips=pending_clips,
                    )
                    pending_clips = []
                pending_destination = parsed_destination
                continue

            clip_file_name = _transfer_clip_file_name(message)
            if not clip_file_name:
                return [], 'External clip(s)'

            route_text = message.caption if message.caption is not None else message.text
            if route_text is not None:
                if _is_sub_group_route_text(route_text):
                    return [], 'Physical group required'
                parsed_destination = _parse_and_validate_destination(
                    route_text, today=today, allowed_years=allowed_years
                )
                if parsed_destination is not None:
                    if pending_clips:
                        _append_transfer_batch(
                            batches,
                            destination_group=pending_destination,
                            clips=pending_clips,
                        )
                        pending_clips = []
                    pending_destination = parsed_destination
            if pending_destination is None:
                return [], 'Invalid input'

            try:
                source_group, clip_id = parse_clip_identity_filename(clip_file_name)
            except ValueError:
                return [], 'External clip(s)'
            pending_clips.append(TransferClipRef(source_group=source_group, clip_id=clip_id))

        if pending_clips:
            _append_transfer_batch(
                batches,
                destination_group=pending_destination,
                clips=pending_clips,
            )
            current_destination = pending_destination
            continue
        if pending_destination is not None:
            current_destination = pending_destination

    if not batches:
        return [], None

    seen_clip_refs: set[tuple[ClipGroup, ClipId]] = set()
    for batch in batches:
        if not batch.clips:
            return [], 'Invalid input'
        for clip in batch.clips:
            identity = (clip.source_group, clip.clip_id)
            if identity in seen_clip_refs:
                return [], 'Invalid input'
            seen_clip_refs.add(identity)

    return batches, None


def _transfer_clip_file_name(message: Message) -> str | None:
    if message.video is not None:
        return message.video.file_name

    document = getattr(message, 'document', None)
    if document is None:
        return None
    return getattr(document, 'file_name', None)


def _append_transfer_batch(
    batches: list[TransferBatch],
    *,
    destination_group: ClipGroup | None,
    clips: list[TransferClipRef],
) -> None:
    if destination_group is None:
        return
    if not batches or batches[-1].destination_group != destination_group:
        batches.append(TransferBatch(destination_group=destination_group, clips=list(clips)))
        return
    batches[-1].clips.extend(clips)


def _parse_and_validate_destination(
    text: str,
    *,
    today: date,
    allowed_years: set[int],
) -> ClipGroup | None:
    parsed_destination = parse_group_selector_text(
        text,
        allow_sub_season_suffix=False,
    )
    if parsed_destination is None:
        return None
    parsed_group, _parsed_sub_season = parsed_destination
    if parsed_group.year not in allowed_years:
        return None
    if parsed_group.season not in store_allowed_seasons(year=parsed_group.year, today=today):
        return None
    return parsed_group


def _is_sub_group_route_text(text: str) -> bool:
    parsed_destination = parse_group_selector_text(
        text,
        allow_sub_season_suffix=True,
    )
    if parsed_destination is None:
        return False
    _parsed_group, parsed_sub_season = parsed_destination
    return parsed_sub_season.exists
