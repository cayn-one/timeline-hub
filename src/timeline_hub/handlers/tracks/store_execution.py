import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TypeVar
from urllib.parse import parse_qs, urlparse

from aiogram import Bot
from aiogram.types import Message
from loguru import logger

from timeline_hub.infra.ffmpeg import probe_audio_sample_rate, to_opus
from timeline_hub.infra.images import normalize_cover_to_jpg
from timeline_hub.infra.ytdlp import (
    DownloadedAudio,
    TrackMetadata,
    UrlTrackInfo,
    YtDlpAuthenticationError,
    YtDlpMetadataError,
    download_audio_as_opus,
    fetch_track_info,
)
from timeline_hub.services.track_store import (
    Track,
    TrackGroup,
    TrackId,
    TrackInvalidAudioFormatError,
    TrackStore,
    UploadedVariant,
)
from timeline_hub.services.youtube_cookies import YoutubeCookieStore, YoutubeCookieStoreError
from timeline_hub.types import Extension, FileBytes, InvalidExtensionError


class TrackInputError(ValueError):
    pass


class TrackLinkDownloadError(RuntimeError):
    pass


class YoutubeCookieAuthenticationRetryExhaustedError(YtDlpAuthenticationError):
    """Raised when refreshed YouTube cookies are rejected by a second operation attempt."""


class UploadedFileTooBigError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LinkOnlyTrackInput:
    url: str
    artists: tuple[str, ...] | None
    title: str | None

    @property
    def requires_metadata(self) -> bool:
        return self.artists is None or self.title is None


@dataclass(frozen=True, slots=True)
class DownloadedLinkAudioCover:
    audio: FileBytes
    cover: FileBytes
    metadata: TrackMetadata | None


@dataclass(frozen=True, slots=True)
class CoverLinkTrackInput:
    url: str
    artists: tuple[str, ...] | None
    title: str | None

    @property
    def requires_metadata(self) -> bool:
        return self.artists is None or self.title is None


@dataclass(frozen=True, slots=True)
class TrackAudioAttachment:
    file_id: str
    file_name: str | None


@dataclass(frozen=True, slots=True)
class UploadedFileRef:
    """One Telegram file-bearing Uploaded item before logical grouping.

    `logical_name` is the reconstructed filename with any `.partN` transport
    suffix removed. `part_index` is `None` for a normal upload and `0..9` for
    multipart fragments.
    """

    logical_name: str
    file_id: str
    file_name: str
    part_index: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedUploadedVariant:
    """One grouped Uploaded variant logical file with parsed speed metadata."""

    file_refs: tuple[UploadedFileRef, ...]
    speed: float
    reverb: float


_UPLOADED_PART_SUFFIX_PATTERN = re.compile(r'^(?P<logical_name>.+?)\.part(?P<part_index>[0-9])$', re.IGNORECASE)
_SUPPORTED_DOCUMENT_AUDIO_EXTENSIONS = frozenset({'.wav', '.flac', '.opus'})
_SUPPORTED_UPLOADED_LOGICAL_EXTENSIONS = frozenset({Extension.OPUS, Extension.MP3})
_T = TypeVar('_T')


def extract_track_audio_attachment(message: Message) -> TrackAudioAttachment | None:
    audio = message.audio
    if audio is not None:
        return TrackAudioAttachment(file_id=audio.file_id, file_name=audio.file_name)

    document = getattr(message, 'document', None)
    if document is None:
        return None
    file_name = getattr(document, 'file_name', None)
    if not _is_supported_document_audio_filename(file_name):
        return None
    file_id = getattr(document, 'file_id', None)
    if not isinstance(file_id, str) or not file_id:
        return None
    return TrackAudioAttachment(file_id=file_id, file_name=file_name)


def _is_supported_document_audio_filename(file_name: str | None) -> bool:
    if file_name is None:
        return False
    return any(file_name.lower().endswith(suffix) for suffix in _SUPPORTED_DOCUMENT_AUDIO_EXTENSIONS)


def extract_uploaded_file_ref(message: Message) -> UploadedFileRef | None:
    return _extract_uploaded_file_ref(message, accepted_extensions=_SUPPORTED_UPLOADED_LOGICAL_EXTENSIONS)


def extract_uploaded_opus_attachment(message: Message) -> UploadedFileRef | None:
    return _extract_uploaded_file_ref(message, accepted_extensions=frozenset({Extension.OPUS}))


def extract_uploaded_mp3_attachment(message: Message) -> UploadedFileRef | None:
    uploaded_file_ref = _extract_uploaded_file_ref(message, accepted_extensions=frozenset({Extension.MP3}))
    if uploaded_file_ref is None:
        return None
    try:
        parse_uploaded_variant_filename(uploaded_file_ref.logical_name)
    except TrackInputError:
        return None
    return uploaded_file_ref


def _extract_uploaded_file_ref(
    message: Message,
    *,
    accepted_extensions: frozenset[Extension],
) -> UploadedFileRef | None:
    for media in (message.audio, getattr(message, 'document', None)):
        if media is None:
            continue
        file_name = getattr(media, 'file_name', None)
        file_id = getattr(media, 'file_id', None)
        if not isinstance(file_name, str) or not isinstance(file_id, str) or not file_id:
            continue
        parsed_file_name = _parse_uploaded_logical_file_name(file_name)
        if parsed_file_name is None:
            continue
        logical_name, part_index = parsed_file_name
        extension = Extension.try_from_filename(logical_name)
        if extension not in accepted_extensions:
            continue
        return UploadedFileRef(
            logical_name=logical_name,
            file_id=file_id,
            file_name=file_name,
            part_index=part_index,
        )
    return None


def _parse_uploaded_logical_file_name(file_name: str) -> tuple[str, int | None] | None:
    match = _UPLOADED_PART_SUFFIX_PATTERN.fullmatch(file_name)
    if match is not None:
        return match.group('logical_name'), int(match.group('part_index'))
    return file_name, None


def extract_single_photo_audio_messages(messages: Sequence[Message]) -> tuple[Message, Message]:
    """Return exactly one photo message and one audio message, order-independent."""
    if len(messages) != 2:
        raise TrackInputError('Invalid input')

    photo_messages = [message for message in messages if message.photo is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(photo_messages) != 1 or len(audio_messages) != 1:
        raise TrackInputError('Invalid input')
    return photo_messages[0], audio_messages[0]


def extract_photo_messages_for_remove(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Return one or more photo messages for remove actions."""
    if len(messages) < 1:
        raise TrackInputError('Invalid input')
    if any(message.photo is None for message in messages):
        raise TrackInputError('Invalid input')
    return tuple(messages)


def _validate_and_order_uploaded_variant_file_refs(
    uploaded_file_refs: Sequence[UploadedFileRef],
) -> tuple[ParsedUploadedVariant, ...]:
    """Validate grouped uploaded variant file refs fully before any file download."""
    if not uploaded_file_refs:
        raise TrackInputError('Invalid input')

    grouped_file_refs: dict[str, list[UploadedFileRef]] = {}
    for uploaded_file_ref in uploaded_file_refs:
        grouped_file_refs.setdefault(uploaded_file_ref.logical_name, []).append(uploaded_file_ref)

    parsed_variants: list[ParsedUploadedVariant] = []
    for logical_name, file_refs in grouped_file_refs.items():
        ordered_file_refs = _validate_and_order_uploaded_file_refs(file_refs)
        speed, reverb = parse_uploaded_variant_filename(logical_name)
        if not math.isfinite(speed) or speed <= 0.0:
            raise TrackInputError('Invalid input')
        if not math.isfinite(reverb) or reverb < 0.0:
            raise TrackInputError('Invalid input')
        parsed_variants.append(
            ParsedUploadedVariant(
                file_refs=ordered_file_refs,
                speed=speed,
                reverb=reverb,
            )
        )

    if not parsed_variants or len(parsed_variants) > 10:
        raise TrackInputError('Invalid input')

    ordered_variants = sorted(parsed_variants, key=lambda variant: variant.speed)
    previous_speed: float | None = None
    for variant in ordered_variants:
        if previous_speed is not None and variant.speed <= previous_speed:
            raise TrackInputError('Invalid input')
        previous_speed = variant.speed

    return tuple(ordered_variants)


def _validate_and_order_uploaded_file_refs(
    uploaded_file_refs: Sequence[UploadedFileRef],
) -> tuple[UploadedFileRef, ...]:
    if not uploaded_file_refs or len(uploaded_file_refs) > 10:
        raise TrackInputError('Invalid input')

    part_indices = [uploaded_file_ref.part_index for uploaded_file_ref in uploaded_file_refs]
    has_whole_file = any(part_index is None for part_index in part_indices)
    has_multipart_parts = any(part_index is not None for part_index in part_indices)
    if has_whole_file and has_multipart_parts:
        raise TrackInputError('Invalid input')

    if has_whole_file:
        if len(uploaded_file_refs) != 1:
            raise TrackInputError('Invalid input')
        return (uploaded_file_refs[0],)

    def _uploaded_file_ref_part_index(uploaded_file_ref: UploadedFileRef) -> int:
        part_index = uploaded_file_ref.part_index
        if part_index is None:
            raise TrackInputError('Invalid input')
        return part_index

    if len(uploaded_file_refs) < 2:
        raise TrackInputError('Invalid input')

    ordered_file_refs = tuple(sorted(uploaded_file_refs, key=_uploaded_file_ref_part_index))
    expected_part_indices = list(range(len(ordered_file_refs)))
    if [uploaded_file_ref.part_index for uploaded_file_ref in ordered_file_refs] != expected_part_indices:
        raise TrackInputError('Invalid input')

    return ordered_file_refs


def validate_uploaded_source_variant_batch(
    messages: Sequence[Message],
) -> tuple[Message, tuple[UploadedFileRef, ...] | None, tuple[ParsedUploadedVariant, ...]]:
    """Validate one ordered uploaded action batch before any file download.

    Expected shape:
        1. exactly one identity-bearing cover photo
        2. zero or one logical `.opus` source file
        3. one or more logical `.mp3` variant files

    Source and variant file-bearing messages may arrive in any order.
    """
    if len(messages) < 2:
        raise TrackInputError('Invalid input')
    if any(message.text is not None for message in messages):
        raise TrackInputError('Invalid input')
    if any(message.video is not None or getattr(message, 'animation', None) is not None for message in messages):
        raise TrackInputError('Invalid input')

    photo_messages = [message for message in messages if message.photo is not None]
    if len(photo_messages) != 1:
        raise TrackInputError('Invalid input')
    photo_message = photo_messages[0]

    uploaded_file_refs = []
    for message in messages:
        if message.photo is not None:
            continue
        uploaded_file_ref = extract_uploaded_file_ref(message)
        if uploaded_file_ref is None:
            raise TrackInputError('Invalid input')
        uploaded_file_refs.append(uploaded_file_ref)

    grouped_file_refs: dict[str, list[UploadedFileRef]] = {}
    for uploaded_file_ref in uploaded_file_refs:
        grouped_file_refs.setdefault(uploaded_file_ref.logical_name, []).append(uploaded_file_ref)

    source_file_refs: tuple[UploadedFileRef, ...] | None = None
    variant_file_refs: list[UploadedFileRef] = []
    for logical_name, grouped_refs in grouped_file_refs.items():
        ordered_refs = _validate_and_order_uploaded_file_refs(grouped_refs)
        extension = Extension.try_from_filename(logical_name)
        if extension is Extension.OPUS:
            if source_file_refs is not None:
                raise TrackInputError('Invalid input')
            source_file_refs = ordered_refs
            continue
        if extension is Extension.MP3:
            variant_file_refs.extend(ordered_refs)
            continue
        raise TrackInputError('Invalid input')

    parsed_variants = _validate_and_order_uploaded_variant_file_refs(variant_file_refs)
    return photo_message, source_file_refs, parsed_variants


def extract_track_identity_from_photo_message(photo_message: Message) -> tuple[TrackGroup, TrackId]:
    """Decode linked-dot cover caption identity into `(group, track_id)`."""
    caption = photo_message.caption
    if caption is None or not caption or caption[0] != '·':
        raise TrackInputError('Invalid input')

    entities = photo_message.caption_entities
    if not entities:
        raise TrackInputError('Invalid input')

    link_url: str | None = None
    for entity in entities:
        entity_type = getattr(entity, 'type', None)
        if getattr(entity_type, 'value', entity_type) != 'text_link':
            continue
        if getattr(entity, 'offset', None) != 0 or getattr(entity, 'length', None) != 1:
            continue
        link_url = getattr(entity, 'url', None)
        break

    if not link_url or not isinstance(link_url, str):
        raise TrackInputError('Invalid input')
    if not link_url.startswith('https://'):
        raise TrackInputError('Invalid input')

    identity = link_url.removeprefix('https://')
    if identity.endswith('.com/'):
        identity = identity.removesuffix('.com/')
    elif identity.endswith('.com'):
        identity = identity.removesuffix('.com')
    else:
        raise TrackInputError('Invalid input')

    if not identity:
        raise TrackInputError('Invalid input')
    return TrackStore.string_to_track_identity(identity)


async def prepare_audio_from_message(*, bot: Bot, audio_message: Message) -> FileBytes:
    """Download one audio message and normalize to OPUS `FileBytes`."""
    attachment = extract_track_audio_attachment(audio_message)
    if attachment is None:
        raise TrackInputError('Invalid input')

    audio_bytes = await _download_file_bytes(bot=bot, file_id=attachment.file_id)
    try:
        audio_extension = Extension.try_from_filename(attachment.file_name)
        if audio_extension is Extension.OPUS:
            audio_opus = await _validate_raw_opus_audio_bytes(audio_bytes)
        else:
            audio_opus = await to_opus(audio_bytes)
    except TrackInvalidAudioFormatError:
        raise
    except Exception as error:
        raise TrackInputError("Can't process audio") from error

    return FileBytes(data=audio_opus, extension=Extension.OPUS)


async def prepare_uploaded_source_from_file_refs(
    *,
    bot: Bot,
    source_file_refs: Sequence[UploadedFileRef],
    max_file_size_bytes: int,
) -> FileBytes:
    """Download and validate an Uploaded source file without transcoding it."""
    raw_bytes = await _download_uploaded_file_refs(
        bot=bot,
        file_refs=source_file_refs,
        max_file_size_bytes=max_file_size_bytes,
    )
    try:
        audio_opus = await _validate_raw_opus_audio_bytes(raw_bytes)
    except TrackInvalidAudioFormatError:
        raise
    except Exception as error:
        raise TrackInputError("Can't process audio") from error

    return FileBytes(data=audio_opus, extension=Extension.OPUS)


def parse_uploaded_variant_filename(file_name: str) -> tuple[float, float]:
    """Parse exact `<int>_<int>.mp3` metadata from an uploaded variant filename."""
    extension = Extension.try_from_filename(file_name)
    if extension is not Extension.MP3:
        raise TrackInputError('Invalid input')

    stem_parts = Path(file_name).stem.split('_')
    if len(stem_parts) != 2:
        raise TrackInputError('Invalid input')

    try:
        speed_part = int(stem_parts[0])
        reverb_part = int(stem_parts[1])
    except ValueError as error:
        raise TrackInputError('Invalid input') from error
    return speed_part / 100, reverb_part / 10


async def prepare_uploaded_variant_from_parsed(
    *,
    bot: Bot,
    parsed_variant: ParsedUploadedVariant,
    max_file_size_bytes: int,
) -> UploadedVariant:
    """Download one validated uploaded MP3 message without transcoding it."""
    raw_bytes = await _download_uploaded_file_refs(
        bot=bot,
        file_refs=parsed_variant.file_refs,
        max_file_size_bytes=max_file_size_bytes,
    )
    try:
        return UploadedVariant(
            speed=parsed_variant.speed,
            reverb=parsed_variant.reverb,
            audio=FileBytes(data=raw_bytes, extension=Extension.MP3),
        )
    except (InvalidExtensionError, ValueError) as error:
        raise TrackInputError('Invalid input') from error


def extract_store_messages(messages: Sequence[Message]) -> list[Message]:
    """Return store-relevant messages in original order."""
    return [
        message
        for message in messages
        if message.photo is not None or extract_track_audio_attachment(message) is not None
    ]


def track_count_from_store_messages(messages: Sequence[Message]) -> int:
    return len(extract_store_messages(messages)) // 2


def extract_audio_only_store_messages(messages: Sequence[Message]) -> tuple[Message | None, Message]:
    """Return validated audio-only store inputs as optional text + audio."""
    if len(messages) == 0 or len(messages) > 2:
        raise TrackInputError('Invalid input')
    if any(
        message.photo is not None or message.video is not None or getattr(message, 'animation', None) is not None
        for message in messages
    ):
        raise TrackInputError('Invalid input')
    text_messages = [message for message in messages if message.text is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(audio_messages) != 1 or len(text_messages) > 1:
        raise TrackInputError('Invalid input')
    audio_message = audio_messages[0]
    text_message = text_messages[0] if text_messages else None
    if len(messages) == 1:
        if text_message is not None:
            raise TrackInputError('Invalid input')
        if audio_message.caption is None:
            raise TrackInputError('Invalid input')
        return None, audio_message
    if len(messages) != 2 or text_message is None:
        raise TrackInputError('Invalid input')
    if audio_message.caption is not None:
        raise TrackInputError('Invalid input')
    return text_message, audio_message


def parse_audio_only_track_metadata(
    *, text_message: Message | None, audio_message: Message
) -> tuple[tuple[str, ...], str]:
    """Parse artists/title for audio-only store from a plain text message."""
    try:
        source = text_message.text if text_message is not None else audio_message.caption
        return _caption_to_artists_and_title(source)
    except TrackInputError as error:
        raise TrackInputError('Invalid input') from error


def validate_audio_only_store_input(messages: Sequence[Message]) -> tuple[tuple[str, ...], str]:
    """Validate audio-only store shape and metadata without downloading files."""
    text_message, audio_message = extract_audio_only_store_messages(messages)
    return parse_audio_only_track_metadata(text_message=text_message, audio_message=audio_message)


def is_supported_youtube_store_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    normalized_url = url.strip()
    if not normalized_url:
        return False
    parsed = urlparse(normalized_url)
    if parsed.scheme != 'https':
        return False
    if parsed.netloc not in ('www.youtube.com', 'music.youtube.com', 'youtu.be'):
        return False
    if parsed.netloc == 'youtu.be':
        path_segments = [segment for segment in parsed.path.split('/') if segment]
        return len(path_segments) == 1
    if parsed.path != '/watch':
        return False
    video_ids = parse_qs(parsed.query).get('v', [])
    return any(video_id.strip() for video_id in video_ids)


def parse_link_only_store_input(text: str) -> LinkOnlyTrackInput:
    if not isinstance(text, str):
        raise TrackInputError('Invalid input')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise TrackInputError('Invalid input')
    url = lines[0]
    if not is_supported_youtube_store_url(url):
        raise TrackInputError('Invalid input')
    if len(lines) == 1:
        return LinkOnlyTrackInput(url=url, artists=None, title=None)
    if len(lines) == 2:
        raise TrackInputError('Invalid input')
    artists = tuple(lines[1:-1])
    title = lines[-1]
    if not artists or any(not artist for artist in artists) or not title:
        raise TrackInputError('Invalid input')
    return LinkOnlyTrackInput(url=url, artists=artists, title=title)


def validate_link_only_store_input(messages: Sequence[Message]) -> LinkOnlyTrackInput:
    if len(messages) != 1:
        raise TrackInputError('Invalid input')
    message = messages[0]
    if message.photo is not None or extract_track_audio_attachment(message) is not None:
        raise TrackInputError('Invalid input')
    if message.video is not None or getattr(message, 'animation', None) is not None:
        raise TrackInputError('Invalid input')
    if message.text is None:
        raise TrackInputError('Invalid input')
    return parse_link_only_store_input(message.text)


def extract_audio_link_store_messages(messages: Sequence[Message]) -> tuple[Message, Message]:
    """Return validated URL text + audio messages, order-independent."""
    if len(messages) != 2:
        raise TrackInputError('Invalid input')
    if any(
        message.photo is not None or message.video is not None or getattr(message, 'animation', None) is not None
        for message in messages
    ):
        raise TrackInputError('Invalid input')

    text_messages = [message for message in messages if message.text is not None]
    audio_messages = [message for message in messages if extract_track_audio_attachment(message) is not None]
    if len(text_messages) != 1 or len(audio_messages) != 1:
        raise TrackInputError('Invalid input')
    text_message = text_messages[0]
    audio_message = audio_messages[0]
    if audio_message.caption is not None:
        raise TrackInputError('Invalid input')
    return text_message, audio_message


def validate_audio_link_store_input(messages: Sequence[Message]) -> tuple[Message, LinkOnlyTrackInput]:
    """Validate one URL text + audio input and parse URL metadata lines."""
    text_message, audio_message = extract_audio_link_store_messages(messages)
    if text_message.text is None:
        raise TrackInputError('Invalid input')
    return audio_message, parse_link_only_store_input(text_message.text)


def parse_cover_link_store_input(messages: Sequence[Message]) -> CoverLinkTrackInput:
    if len(messages) != 1:
        raise TrackInputError('Invalid input')

    message = messages[0]
    if message.photo is None:
        raise TrackInputError('Invalid input')
    if (
        extract_track_audio_attachment(message) is not None
        or message.video is not None
        or getattr(message, 'animation', None) is not None
    ):
        raise TrackInputError('Invalid input')
    if message.caption is None:
        raise TrackInputError('Invalid input')
    parsed_link_input = parse_link_only_store_input(message.caption)

    return CoverLinkTrackInput(
        url=parsed_link_input.url,
        artists=parsed_link_input.artists,
        title=parsed_link_input.title,
    )


async def fetch_link_track_info(
    url: str,
    *,
    cookie_store: YoutubeCookieStore,
    with_cover: bool,
    with_metadata: bool,
) -> UrlTrackInfo:
    return await _run_youtube_operation_with_cookie_refresh(
        cookie_store=cookie_store,
        operation=lambda cookie_file: fetch_track_info(
            url,
            cookie_file=cookie_file,
            with_cover=with_cover,
            with_metadata=with_metadata,
        ),
    )


async def download_link_audio(
    url: str,
    *,
    cookie_store: YoutubeCookieStore,
    max_duration: timedelta | None = None,
) -> FileBytes:
    try:
        result = await _run_youtube_operation_with_cookie_refresh(
            cookie_store=cookie_store,
            operation=lambda cookie_file: download_audio_as_opus(
                url,
                cookie_file=cookie_file,
                max_duration=max_duration,
            ),
        )
    except YoutubeCookieAuthenticationRetryExhaustedError, YoutubeCookieStoreError:
        raise
    except Exception as error:
        raise TrackLinkDownloadError(str(error)) from error
    return FileBytes(data=result.audio, extension=Extension.OPUS)


async def download_link_audio_and_cover(
    url: str,
    *,
    cookie_store: YoutubeCookieStore,
    with_metadata: bool = False,
    max_duration: timedelta | None = None,
) -> DownloadedLinkAudioCover:
    try:
        result = await _run_youtube_operation_with_cookie_refresh(
            cookie_store=cookie_store,
            operation=lambda cookie_file: download_audio_as_opus(
                url,
                cookie_file=cookie_file,
                with_cover=True,
                with_metadata=with_metadata,
                max_duration=max_duration,
            ),
        )
    except YtDlpMetadataError, YoutubeCookieAuthenticationRetryExhaustedError, YoutubeCookieStoreError:
        raise
    except Exception as error:
        raise TrackLinkDownloadError(str(error)) from error
    if result.cover is None:
        raise TrackLinkDownloadError('yt-dlp did not produce cover output')
    return DownloadedLinkAudioCover(
        audio=FileBytes(data=result.audio, extension=Extension.OPUS),
        cover=FileBytes(data=result.cover, extension=Extension.JPG),
        metadata=result.metadata,
    )


async def download_link_audio_with_metadata(
    url: str,
    *,
    cookie_store: YoutubeCookieStore,
    with_metadata: bool,
    max_duration: timedelta | None,
) -> DownloadedAudio:
    return await _run_youtube_operation_with_cookie_refresh(
        cookie_store=cookie_store,
        operation=lambda cookie_file: download_audio_as_opus(
            url,
            cookie_file=cookie_file,
            with_metadata=with_metadata,
            max_duration=max_duration,
        ),
    )


async def _run_youtube_operation_with_cookie_refresh(
    *,
    cookie_store: YoutubeCookieStore,
    operation: Callable[[Path], Awaitable[_T]],
) -> _T:
    snapshot = cookie_store.current()
    try:
        return await operation(snapshot.path)
    except YtDlpAuthenticationError:
        logger.warning('yt-dlp authentication rejected; retrying with latest YouTube cookies')
        refreshed_snapshot = await cookie_store.refresh_after_rejection(snapshot)

    try:
        return await operation(refreshed_snapshot.path)
    except YtDlpAuthenticationError as error:
        logger.error('yt-dlp authentication rejected after YouTube cookie refresh')
        raise YoutubeCookieAuthenticationRetryExhaustedError('YouTube cookies were rejected after refresh') from error


def validate_track_batch(messages: Sequence[Message]) -> list[tuple[tuple[str, ...], str]]:
    if len(messages) < 2 or len(messages) % 2 != 0:
        raise TrackInputError("Can't dispatch input")

    parsed_tracks: list[tuple[tuple[str, ...], str]] = []
    for index in range(0, len(messages), 2):
        photo_message = messages[index]
        audio_message = messages[index + 1]
        if photo_message.photo is None or extract_track_audio_attachment(audio_message) is None:
            raise TrackInputError("Can't dispatch input")
        if photo_message.caption is None or not photo_message.caption.strip():
            raise TrackInputError("Can't dispatch input")

        try:
            parsed_tracks.append(_caption_to_artists_and_title(photo_message.caption))
        except TrackInputError as error:
            raise TrackInputError("Can't dispatch input") from error

    return parsed_tracks


async def prepare_tracks_from_buffer(*, bot: Bot, messages: Sequence[Message]) -> list[Track]:
    store_messages = extract_store_messages(messages)
    parsed_tracks = validate_track_batch(store_messages)
    prepared_tracks: list[Track] = []
    for parsed_track, index in zip(parsed_tracks, range(0, len(store_messages), 2), strict=True):
        photo_message = store_messages[index]
        audio_message = store_messages[index + 1]
        photo = photo_message.photo
        attachment = extract_track_audio_attachment(audio_message)
        if photo is None or attachment is None:
            raise TrackInputError("Can't dispatch input")

        artists, title = parsed_track
        cover_bytes = await _download_file_bytes(
            bot=bot,
            file_id=photo[-1].file_id,
        )
        audio_bytes = await _download_file_bytes(
            bot=bot,
            file_id=attachment.file_id,
        )

        try:
            cover_jpg = normalize_cover_to_jpg(cover_bytes)
        except Exception as error:
            raise TrackInputError("Can't process cover image") from error

        try:
            # Best-effort extension parse (filename may be missing or invalid).
            audio_extension = Extension.try_from_filename(attachment.file_name)
            if audio_extension is Extension.OPUS:
                # Fast-path: avoid re-encoding already-Opus input.
                audio_opus = await _validate_raw_opus_audio_bytes(audio_bytes)
            else:
                audio_opus = await to_opus(audio_bytes)
        except TrackInvalidAudioFormatError:
            raise
        except Exception as error:
            raise TrackInputError("Can't process audio") from error

        prepared_tracks.append(
            Track(
                artists=artists,
                title=title,
                cover=FileBytes(data=cover_jpg, extension=Extension.JPG),
                audio=FileBytes(data=audio_opus, extension=Extension.OPUS),
            )
        )

    return prepared_tracks


async def _validate_raw_opus_audio_bytes(audio_bytes: bytes) -> bytes:
    """Validate raw Opus source sample rate without re-encoding valid input."""
    sample_rate = await probe_audio_sample_rate(audio_bytes)
    if sample_rate != 48_000:
        raise TrackInvalidAudioFormatError(f'Audio sample rate must be 48000 Hz, got {sample_rate}')
    return audio_bytes


async def prepare_audio_only_track_from_buffer(
    *,
    bot: Bot,
    messages: Sequence[Message],
    album_id: str,
) -> Track:
    """Prepare one store-ready track for audio-only + text metadata flows."""
    text_message, audio_message = extract_audio_only_store_messages(messages)
    artists, title = parse_audio_only_track_metadata(text_message=text_message, audio_message=audio_message)
    audio = await prepare_audio_from_message(bot=bot, audio_message=audio_message)
    return Track(
        artists=artists,
        title=title,
        audio=audio,
        cover=None,
        album_id=album_id,
    )


def prepare_link_only_track_from_buffer(
    *,
    messages: Sequence[Message],
) -> LinkOnlyTrackInput:
    return validate_link_only_store_input(messages)


def prepare_audio_link_track_from_buffer(
    *,
    messages: Sequence[Message],
) -> tuple[Message, LinkOnlyTrackInput]:
    return validate_audio_link_store_input(messages)


def _caption_to_artists_and_title(caption: str | None) -> tuple[tuple[str, ...], str]:
    lines = [line.strip() for line in (caption or '').splitlines() if line.strip()]
    if len(lines) < 2:
        raise TrackInputError('Not enough lines to extract artists and title')
    return tuple(lines[:-1]), lines[-1]


async def _download_file_bytes(*, bot: Bot, file_id: str) -> bytes:
    telegram_file = await bot.get_file(file_id)
    if telegram_file.file_path is None:
        raise TrackInputError("Can't dispatch input")

    downloaded = await bot.download_file(telegram_file.file_path)
    if downloaded is None:
        raise TrackInputError("Can't dispatch input")

    return downloaded.read()


async def _download_uploaded_file_refs(
    *,
    bot: Bot,
    file_refs: Sequence[UploadedFileRef],
    max_file_size_bytes: int,
) -> bytes:
    if not file_refs:
        raise TrackInputError('Invalid input')
    ordered_file_refs = sorted(
        file_refs,
        key=lambda uploaded_file_ref: 0 if uploaded_file_ref.part_index is None else uploaded_file_ref.part_index,
    )
    downloaded_parts = [
        await _download_file_bytes(bot=bot, file_id=uploaded_file_ref.file_id)
        for uploaded_file_ref in ordered_file_refs
    ]
    raw_bytes = downloaded_parts[0] if len(downloaded_parts) == 1 else b''.join(downloaded_parts)
    if len(raw_bytes) > max_file_size_bytes:
        raise UploadedFileTooBigError('Uploaded file exceeds Telegram media group size limit')
    return raw_bytes
