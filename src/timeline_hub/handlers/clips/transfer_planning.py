from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from aiogram.types import Message

from timeline_hub.handlers.clips.common import extract_clip_file_id
from timeline_hub.handlers.clips.flow import store_allowed_seasons, year_option_universe
from timeline_hub.handlers.clips.reconcile_input import (
    ChronologicalClipRef,
    ParsedClipRef,
    parse_clip_identity_filename,
)
from timeline_hub.handlers.clips.route_planning import parse_inbox_route_text, parse_route_text
from timeline_hub.services.clip_store import ClipGroup, ClipId, ClipSubGroup, Scope, SubSeason, Universe
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


@dataclass(slots=True)
class InboxTransferBatch:
    source_universe: Universe
    destination_group: ClipGroup
    destination_sub_group: ClipSubGroup
    clip_ids: list[ClipId]


type PlannedTransferBatch = TransferBatch | InboxTransferBatch
type _TransferDestination = tuple[ClipGroup, SubSeason] | Universe


def plan_transfer_batches(
    message_groups: Sequence[MessageGroup],
    *,
    settings: Settings,
    resolved_clip_refs: Mapping[int, ParsedClipRef] | None = None,
) -> tuple[list[PlannedTransferBatch], str | None]:
    """Plan authoritative chronological and inbox clips into ordered transfers."""
    batches: list[PlannedTransferBatch] = []
    current_destination: _TransferDestination | None = None
    today = date.today()
    allowed_years = set(year_option_universe(current_year=today.year, min_year=settings.min_year))

    for message_group in message_groups:
        for message in message_group:
            if extract_clip_file_id(message) is None:
                if message.text is None:
                    continue
                parsed_destination = _parse_and_validate_route_destination(
                    message.text,
                    today=today,
                    allowed_years=allowed_years,
                )
                if parsed_destination is not None:
                    current_destination = parsed_destination
                continue

            clip_file_name = _transfer_clip_file_name(message)
            if not clip_file_name:
                return [], 'External clip(s)'
            route_text = message.caption if message.caption is not None else message.text
            if route_text is not None:
                parsed_destination = _parse_and_validate_route_destination(
                    route_text,
                    today=today,
                    allowed_years=allowed_years,
                )
                if parsed_destination is not None:
                    current_destination = parsed_destination
            if current_destination is None:
                return [], 'Invalid input'
            if isinstance(current_destination, Universe):
                return [], 'Inbox routes require external clips'

            source_ref = None if resolved_clip_refs is None else resolved_clip_refs.get(message.message_id)
            if source_ref is None:
                try:
                    source_ref = parse_clip_identity_filename(clip_file_name)
                except ValueError:
                    return [], 'External clip(s)'
            destination_group, destination_sub_season = current_destination
            if isinstance(source_ref, ChronologicalClipRef):
                if destination_sub_season.exists:
                    return [], 'Physical group required'
                _append_transfer_batch(
                    batches,
                    destination_group=destination_group,
                    clip=TransferClipRef(source_group=source_ref.group, clip_id=source_ref.clip_id),
                )
                continue

            _append_inbox_transfer_batch(
                batches,
                source_universe=source_ref.universe,
                destination_group=destination_group,
                destination_sub_group=ClipSubGroup(
                    sub_season=destination_sub_season,
                    scope=Scope.SOURCE,
                ),
                clip_id=source_ref.clip_id,
            )

    if not batches:
        return [], None

    seen_clip_refs: set[tuple[object, ClipId]] = set()
    for batch in batches:
        if isinstance(batch, TransferBatch):
            if not batch.clips:
                return [], 'Invalid input'
            identities = ((clip.source_group, clip.clip_id) for clip in batch.clips)
        else:
            if not batch.clip_ids:
                return [], 'Invalid input'
            identities = ((batch.source_universe, clip_id) for clip_id in batch.clip_ids)
        for identity in identities:
            if identity in seen_clip_refs:
                return [], 'Invalid input'
            seen_clip_refs.add(identity)
    return batches, None


def _transfer_clip_file_name(message: Message) -> str | None:
    if message.video is not None:
        return message.video.file_name
    document = getattr(message, 'document', None)
    return None if document is None else getattr(document, 'file_name', None)


def _append_transfer_batch(
    batches: list[PlannedTransferBatch],
    *,
    destination_group: ClipGroup,
    clip: TransferClipRef,
) -> None:
    if batches and isinstance(batches[-1], TransferBatch) and batches[-1].destination_group == destination_group:
        batches[-1].clips.append(clip)
        return
    batches.append(TransferBatch(destination_group=destination_group, clips=[clip]))


def _append_inbox_transfer_batch(
    batches: list[PlannedTransferBatch],
    *,
    source_universe: Universe,
    destination_group: ClipGroup,
    destination_sub_group: ClipSubGroup,
    clip_id: ClipId,
) -> None:
    if (
        batches
        and isinstance(batches[-1], InboxTransferBatch)
        and batches[-1].source_universe is source_universe
        and batches[-1].destination_group == destination_group
        and batches[-1].destination_sub_group == destination_sub_group
    ):
        batches[-1].clip_ids.append(clip_id)
        return
    batches.append(
        InboxTransferBatch(
            source_universe=source_universe,
            destination_group=destination_group,
            destination_sub_group=destination_sub_group,
            clip_ids=[clip_id],
        )
    )


def _parse_and_validate_route_destination(
    text: str,
    *,
    today: date,
    allowed_years: set[int],
) -> _TransferDestination | None:
    inbox_universe = parse_inbox_route_text(text)
    if inbox_universe is not None:
        return inbox_universe
    parsed_destination = parse_route_text(text)
    if parsed_destination is None:
        return None
    group, _sub_season = parsed_destination
    if group.year not in allowed_years:
        return None
    if group.season not in store_allowed_seasons(year=group.year, today=today):
        return None
    return parsed_destination
