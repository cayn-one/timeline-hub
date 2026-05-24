import math
from datetime import date, timedelta
from enum import StrEnum, auto
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.formatting import Bold, Text
from loguru import logger

from timeline_hub.handlers.menu import (
    back_button,
    callback_message,
    create_padding_line,
    fixed_option_keyboard,
    handle_stale_selection,
    selected_text,
    selection_keyboard,
    selection_text,
    validate_flow_state,
)
from timeline_hub.handlers.tracks.retrieve import send_fetched_tracks_by_ids
from timeline_hub.handlers.tracks.store_execution import (
    CoverLinkTrackInput,
    LinkOnlyTrackInput,
    TrackInputError,
    TrackLinkDownloadError,
    download_link_audio,
    download_link_audio_and_cover,
    extract_photo_messages_for_remove,
    extract_single_photo_audio_messages,
    extract_store_messages,
    extract_track_audio_attachment,
    extract_track_identity_from_photo_message,
    is_supported_youtube_store_url,
    parse_cover_link_store_input,
    prepare_audio_from_message,
    prepare_audio_only_track_from_buffer,
    prepare_link_only_track_from_buffer,
    prepare_tracks_from_buffer,
    validate_audio_only_store_input,
    validate_link_only_store_input,
    validate_track_batch,
)
from timeline_hub.infra.images import normalize_cover_to_jpg
from timeline_hub.infra.ytdlp import YtDlpMetadataError, download_audio_as_opus
from timeline_hub.services.container import Services
from timeline_hub.services.track_store import (
    InvalidTrackIdentityError,
    Season,
    SubSeason,
    Track,
    TrackGroup,
    TrackGroupNotFoundError,
    TrackInfo,
    TrackInvalidAudioFormatError,
    TrackManifestCorruptedError,
    TrackPresetsCorruptedError,
    TrackRemoveManifestSyncError,
    TrackUniverse,
    TrackUpdateManifestSyncError,
)
from timeline_hub.settings import Settings
from timeline_hub.types import Extension, FileBytes, InvalidExtensionError

router = Router()
_TRACK_STORE_MODE = 'track_store'
_TRACK_BACK_VALUE = 'back'
_TRACK_COVER_SOURCE_AUTO = 'auto'
_TRACK_COVER_SOURCE_ALBUM = 'album'
_TRACK_STORE_FINAL_SKIP = 'skip'
_TRACK_STORE_FINAL_FETCH = 'fetch'
_EXPECTED_TRACK_METADATA_UNAVAILABLE_ERRORS = frozenset(
    {
        'yt-dlp produced incomplete metadata output',
        'yt-dlp did not produce metadata output',
    }
)
_YOUTUBE_COOKIES_REQUIRED_ERROR_SUBSTRINGS = frozenset(
    {
        'sign in to confirm your age',
        'use --cookies-from-browser or --cookies',
    }
)


class TrackIntakeAction(StrEnum):
    STORE = auto()
    REPLACE = auto()
    REMOVE = auto()
    TRACK = auto()
    INSTRUMENTAL = auto()
    REMOVE_TRACK = auto()
    REMOVE_INSTRUMENTAL = auto()
    BACK = auto()
    CANCEL = auto()


class TrackIntakeActionCallbackData(CallbackData, prefix='track_intake'):
    action: TrackIntakeAction
    buffer_version: int


class TrackStoreAction(StrEnum):
    SELECT = auto()
    BACK = auto()


class TrackStoreCompletionMode(StrEnum):
    SKIP = auto()
    FETCH = auto()


class TrackStoreStep(StrEnum):
    UNIVERSE = auto()
    YEAR = auto()
    SEASON = auto()
    SUB_SEASON = auto()
    COVER_SOURCE = auto()
    ALBUM = auto()
    FINAL = auto()


class TrackStoreCallbackData(CallbackData, prefix='track_store'):
    action: TrackStoreAction
    step: TrackStoreStep
    value: str


def _is_expected_track_metadata_unavailable(error: YtDlpMetadataError) -> bool:
    return str(error) in _EXPECTED_TRACK_METADATA_UNAVAILABLE_ERRORS


def _is_youtube_cookies_required_error(error: TrackLinkDownloadError | Exception) -> bool:
    normalized_error_text = str(error).strip().lower()
    if not normalized_error_text:
        return False
    return any(candidate in normalized_error_text for candidate in _YOUTUBE_COOKIES_REQUIRED_ERROR_SUBSTRINGS)


class TrackStoreFlow(StatesGroup):
    universe = State()
    year = State()
    season = State()
    sub_season = State()
    cover_source = State()
    album = State()
    final = State()


@router.callback_query(
    TrackIntakeActionCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_track_intake_action(
    callback: CallbackQuery,
    callback_data: TrackIntakeActionCallbackData,
    state: FSMContext,
    services: Services,
    settings: Settings,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.buffer_version != services.chat_message_buffer.version(message.chat.id):
        await handle_stale_selection(message=message, state=state)
        return

    if callback_data.action is TrackIntakeAction.CANCEL:
        await state.clear()
        await message.edit_text(
            **selected_text(selected=['Cancel']),
            reply_markup=None,
        )
        services.chat_message_buffer.flush(message.chat.id)
        return

    if callback_data.action is TrackIntakeAction.REPLACE:
        await message.edit_text(
            **_track_replace_menu_kwargs(
                message_width=settings.message_width,
                buffer_version=callback_data.buffer_version,
                message_count=len(services.chat_message_buffer.peek_flat(message.chat.id)),
            )
        )
        return

    if callback_data.action is TrackIntakeAction.REMOVE:
        await message.edit_text(
            **_track_remove_menu_kwargs(
                message_width=settings.message_width,
                buffer_version=callback_data.buffer_version,
            )
        )
        return

    if callback_data.action is TrackIntakeAction.BACK:
        await message.edit_text(
            **_track_intake_menu_kwargs(
                message_width=settings.message_width,
                buffer_version=callback_data.buffer_version,
                message_count=len(services.chat_message_buffer.peek_flat(message.chat.id)),
            )
        )
        return

    if callback_data.action in (TrackIntakeAction.TRACK, TrackIntakeAction.INSTRUMENTAL):
        bot = getattr(callback, 'bot', None)
        if bot is None:
            await _invalidate_track_intake_buffer(message=message, state=state, services=services, text='Invalid input')
            return
        await _execute_track_update(
            message=message,
            state=state,
            services=services,
            action=callback_data.action,
            bot=bot,
            expected_buffer_version=callback_data.buffer_version,
        )
        return

    if callback_data.action in (TrackIntakeAction.REMOVE_TRACK, TrackIntakeAction.REMOVE_INSTRUMENTAL):
        await _execute_track_remove(
            message=message,
            state=state,
            services=services,
            action=callback_data.action,
            expected_buffer_version=callback_data.buffer_version,
        )
        return

    if callback_data.action is TrackIntakeAction.STORE:
        buffered_messages = services.chat_message_buffer.peek_flat(message.chat.id)
        try:
            _validate_store_input_at_entry(buffered_messages)
        except TrackInputError:
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Invalid input',
            )
            return

    await _show_store_universe_menu(
        message=message,
        state=state,
        services=services,
        settings=settings,
    )


@router.callback_query(
    TrackStoreCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_track_store_menu(
    callback: CallbackQuery,
    callback_data: TrackStoreCallbackData,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if not await _validate_track_store_callback(
        message=message,
        state=state,
        services=services,
        step=callback_data.step,
    ):
        return

    data = await state.get_data()
    if callback_data.action is TrackStoreAction.BACK:
        await _handle_track_store_back(
            message=message,
            state=state,
            services=services,
            settings=settings,
            step=callback_data.step,
            data=data,
        )
        return

    match callback_data.step:
        case TrackStoreStep.UNIVERSE:
            try:
                universe = TrackUniverse(callback_data.value)
            except ValueError:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_store_year_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
            )
        case TrackStoreStep.YEAR:
            universe = _selected_universe(data)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            try:
                year = int(callback_data.value)
            except ValueError:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_store_season_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
                year=year,
            )
        case TrackStoreStep.SEASON:
            selection = _selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            try:
                season = Season(int(callback_data.value))
            except ValueError:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_store_sub_season_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
                year=year,
                season=season,
            )
        case TrackStoreStep.SUB_SEASON:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            try:
                sub_season = SubSeason(callback_data.value)
            except ValueError:
                await handle_stale_selection(message=message, state=state)
                return
            await _on_store_sub_season_selected(
                message=message,
                state=state,
                services=services,
                settings=settings,
                bot=bot,
                sub_season=sub_season,
                universe=universe,
                year=year,
                season=season,
            )
        case TrackStoreStep.COVER_SOURCE:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            sub_season = _selected_sub_season(data)
            if sub_season is None:
                await handle_stale_selection(message=message, state=state)
                return
            if callback_data.value == _TRACK_COVER_SOURCE_AUTO:
                await _show_store_final_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    from_step=TrackStoreStep.COVER_SOURCE,
                )
                return
            if callback_data.value == _TRACK_COVER_SOURCE_ALBUM:
                await _show_store_album_menu(
                    message=message,
                    state=state,
                    services=services,
                    settings=settings,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    buffer_version=_buffer_version_from_state(data),
                )
                return
            await handle_stale_selection(message=message, state=state)
        case TrackStoreStep.ALBUM:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            sub_season = _selected_sub_season(data)
            if sub_season is None:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_store_final_menu(
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                from_step=TrackStoreStep.ALBUM,
                album_id=callback_data.value,
            )
        case TrackStoreStep.FINAL:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            sub_season = _selected_sub_season(data)
            if sub_season is None:
                await handle_stale_selection(message=message, state=state)
                return
            completion_mode = _parse_track_store_completion_mode(callback_data.value)
            if completion_mode is None:
                await handle_stale_selection(message=message, state=state)
                return
            await _execute_track_store_final_selection(
                message=message,
                state=state,
                services=services,
                settings=settings,
                bot=bot,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                completion_mode=completion_mode,
                data=data,
            )


async def try_dispatch_track_intake(
    *,
    message: Message,
    services: Services,
    settings: Settings,
) -> bool:
    await message.answer(
        **_track_intake_menu_kwargs(
            message_width=settings.message_width,
            buffer_version=services.chat_message_buffer.version(message.chat.id),
            message_count=len(services.chat_message_buffer.peek_flat(message.chat.id)),
        )
    )
    return True


def _track_intake_menu_kwargs(
    *,
    message_width: int,
    buffer_version: int,
    message_count: int,
) -> dict[str, Any]:
    return {
        **Text(
            create_padding_line(message_width),
            '\n',
            Text('Messages: ', Bold(str(message_count))),
            '. Select action:',
        ).as_kwargs(),
        'reply_markup': selection_keyboard(
            buttons=[
                InlineKeyboardButton(
                    text='Store',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.STORE,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
                InlineKeyboardButton(
                    text='Replace',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.REPLACE,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
                InlineKeyboardButton(
                    text='Remove',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.REMOVE,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
            ],
            back_button=InlineKeyboardButton(
                text='Cancel',
                callback_data=TrackIntakeActionCallbackData(
                    action=TrackIntakeAction.CANCEL,
                    buffer_version=buffer_version,
                ).pack(),
            ),
        ),
    }


def _track_replace_menu_kwargs(
    *,
    message_width: int,
    buffer_version: int,
    message_count: int,
) -> dict[str, Any]:
    return {
        **Text(
            create_padding_line(message_width),
            '\n',
            'Selected: ',
            Bold('Replace'),
        ).as_kwargs(),
        'reply_markup': selection_keyboard(
            buttons=[
                InlineKeyboardButton(
                    text='Track',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.TRACK,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
                InlineKeyboardButton(
                    text='Instrumental',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.INSTRUMENTAL,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
            ],
            back_button=InlineKeyboardButton(
                text='Back',
                callback_data=TrackIntakeActionCallbackData(
                    action=TrackIntakeAction.BACK,
                    buffer_version=buffer_version,
                ).pack(),
            ),
        ),
    }


def _track_remove_menu_kwargs(
    *,
    message_width: int,
    buffer_version: int,
) -> dict[str, Any]:
    return {
        **Text(
            create_padding_line(message_width),
            '\n',
            'Selected: ',
            Bold('Remove'),
        ).as_kwargs(),
        'reply_markup': selection_keyboard(
            buttons=[
                InlineKeyboardButton(
                    text='Track',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.REMOVE_TRACK,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
                InlineKeyboardButton(
                    text='Instrumental',
                    callback_data=TrackIntakeActionCallbackData(
                        action=TrackIntakeAction.REMOVE_INSTRUMENTAL,
                        buffer_version=buffer_version,
                    ).pack(),
                ),
            ],
            back_button=InlineKeyboardButton(
                text='Back',
                callback_data=TrackIntakeActionCallbackData(
                    action=TrackIntakeAction.BACK,
                    buffer_version=buffer_version,
                ).pack(),
            ),
        ),
    }


async def _show_track_intake_action_menu(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
) -> None:
    await state.clear()
    await message.edit_text(
        **_track_intake_menu_kwargs(
            message_width=settings.message_width,
            buffer_version=services.chat_message_buffer.version(message.chat.id),
            message_count=len(services.chat_message_buffer.peek_flat(message.chat.id)),
        )
    )


async def _show_store_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
) -> None:
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.universe,
        menu_message_id=message.message_id,
        buffer_version=services.chat_message_buffer.version(message.chat.id),
    )
    await message.edit_text(
        **selection_text(
            selected=['Store'],
            prompt='Select universe:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(TrackUniverse),
            available_options=tuple(TrackUniverse),
            build_button=lambda universe: InlineKeyboardButton(
                text=_format_universe(universe),
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.SELECT,
                    step=TrackStoreStep.UNIVERSE,
                    value=universe.value,
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.BACK,
                    step=TrackStoreStep.UNIVERSE,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )


async def _show_store_year_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    buffer_version: int,
    universe: TrackUniverse,
) -> None:
    year_options = list(range(date.today().year, settings.min_clip_year - 1, -1))
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.year,
        menu_message_id=message.message_id,
        buffer_version=buffer_version,
        universe=universe,
    )
    await message.edit_text(
        **selection_text(
            selected=['Store', _format_universe(universe)],
            prompt='Select year:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=year_options,
            available_options=year_options,
            build_button=lambda year: InlineKeyboardButton(
                text=str(year),
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.SELECT,
                    step=TrackStoreStep.YEAR,
                    value=str(year),
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.BACK,
                    step=TrackStoreStep.YEAR,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )


async def _show_store_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    buffer_version: int,
    universe: TrackUniverse,
    year: int,
) -> None:
    available_seasons = _available_store_seasons(year=year, today=date.today())
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.season,
        menu_message_id=message.message_id,
        buffer_version=buffer_version,
        universe=universe,
        year=year,
    )
    await message.edit_text(
        **selection_text(
            selected=['Store', _format_universe(universe), str(year)],
            prompt='Select season:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(Season),
            available_options=available_seasons,
            build_button=lambda season: InlineKeyboardButton(
                text=str(int(season)),
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.SELECT,
                    step=TrackStoreStep.SEASON,
                    value=str(int(season)),
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.BACK,
                    step=TrackStoreStep.SEASON,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )


async def _show_store_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    buffer_version: int,
    universe: TrackUniverse,
    year: int,
    season: Season,
) -> None:
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.sub_season,
        menu_message_id=message.message_id,
        buffer_version=buffer_version,
        universe=universe,
        year=year,
        season=season,
        sub_season=None,
    )
    await message.edit_text(
        **selection_text(
            selected=['Store', _format_universe(universe), str(year), str(int(season))],
            prompt='Select sub-season:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(SubSeason),
            available_options=tuple(SubSeason),
            build_button=lambda sub_season: InlineKeyboardButton(
                text=_format_sub_season(sub_season),
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.SELECT,
                    step=TrackStoreStep.SUB_SEASON,
                    value=sub_season.value,
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.BACK,
                    step=TrackStoreStep.SUB_SEASON,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )


async def _show_store_album_menu(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    buffer_version: int,
) -> None:
    group = TrackGroup(universe=universe, year=year, season=season)
    tracks_by_sub_season = await services.track_store.list_tracks(group)
    sub_season_tracks = tracks_by_sub_season.get(sub_season, [])
    album_options = _unique_album_options(sub_season_tracks)
    if not album_options:
        raise TrackInputError('Invalid input')

    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.album,
        menu_message_id=message.message_id,
        buffer_version=buffer_version,
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
    )
    await message.edit_text(
        **selection_text(
            selected=[
                *_selected_store_path(
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                ),
                'Album',
            ],
            prompt='Select album:',
            message_width=settings.message_width,
        ),
        reply_markup=_album_selection_keyboard(album_options=album_options),
    )


async def _show_store_cover_source_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    buffer_version: int,
) -> None:
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.cover_source,
        menu_message_id=message.message_id,
        buffer_version=buffer_version,
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
    )
    await message.edit_text(
        **selection_text(
            selected=_selected_store_path(
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
            ),
            prompt='Select cover source:',
            message_width=settings.message_width,
        ),
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text='Auto',
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.SELECT,
                            step=TrackStoreStep.COVER_SOURCE,
                            value=_TRACK_COVER_SOURCE_AUTO,
                        ).pack(),
                    )
                ],
                [
                    InlineKeyboardButton(
                        text='Album',
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.SELECT,
                            step=TrackStoreStep.COVER_SOURCE,
                            value=_TRACK_COVER_SOURCE_ALBUM,
                        ).pack(),
                    ),
                ],
                [
                    back_button(
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.BACK,
                            step=TrackStoreStep.COVER_SOURCE,
                            value=_TRACK_BACK_VALUE,
                        ).pack(),
                    )
                ],
            ]
        ),
    )


async def _show_store_final_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    from_step: TrackStoreStep,
    album_id: str | None = None,
) -> None:
    selected = _selected_store_path(
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
    )
    if from_step is TrackStoreStep.COVER_SOURCE:
        selected.append('Auto')
    if from_step is TrackStoreStep.ALBUM:
        selected.append('Album')
    await _set_track_store_context(
        state=state,
        fsm_state=TrackStoreFlow.final,
        menu_message_id=message.message_id,
        buffer_version=_buffer_version_from_state(await state.get_data()),
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
    )
    await state.update_data(
        final_from_step=from_step.value,
        final_album_id=album_id,
    )
    await message.edit_text(
        **selection_text(
            selected=selected,
            prompt='Select finish mode:',
            message_width=settings.message_width,
        ),
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text='Skip',
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.SELECT,
                            step=TrackStoreStep.FINAL,
                            value=_TRACK_STORE_FINAL_SKIP,
                        ).pack(),
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text='Fetch',
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.SELECT,
                            step=TrackStoreStep.FINAL,
                            value=_TRACK_STORE_FINAL_FETCH,
                        ).pack(),
                    ),
                ],
                [
                    back_button(
                        callback_data=TrackStoreCallbackData(
                            action=TrackStoreAction.BACK,
                            step=TrackStoreStep.FINAL,
                            value=_TRACK_BACK_VALUE,
                        ).pack(),
                    )
                ],
            ]
        ),
    )


async def _execute_track_store_final_selection(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    completion_mode: TrackStoreCompletionMode,
    data: dict[str, object],
) -> None:
    final_from_step = _final_from_step(data)
    if final_from_step is None:
        await handle_stale_selection(message=message, state=state)
        return
    if final_from_step is TrackStoreStep.SUB_SEASON:
        await _on_store_sub_season_selected_with_mode(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            sub_season=sub_season,
            universe=universe,
            year=year,
            season=season,
            completion_mode=completion_mode,
        )
        return
    if final_from_step is TrackStoreStep.COVER_SOURCE:
        await _execute_track_store_with_auto_cover(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            universe=universe,
            year=year,
            season=season,
            sub_season=sub_season,
            completion_mode=completion_mode,
        )
        return
    if final_from_step is TrackStoreStep.ALBUM:
        album_id = _final_album_id(data)
        if album_id is None:
            await handle_stale_selection(message=message, state=state)
            return
        await _execute_track_store_with_album_reuse(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            universe=universe,
            year=year,
            season=season,
            sub_season=sub_season,
            album_id=album_id,
            completion_mode=completion_mode,
        )
        return
    await handle_stale_selection(message=message, state=state)


async def _handle_track_store_back(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    step: TrackStoreStep,
    data: dict[str, object],
) -> None:
    match step:
        case TrackStoreStep.UNIVERSE:
            await _show_track_intake_action_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
            )
        case TrackStoreStep.YEAR:
            await _show_store_universe_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
            )
        case TrackStoreStep.SEASON:
            universe = _selected_universe(data)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_store_year_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
            )
        case TrackStoreStep.SUB_SEASON:
            selection = _selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            await _show_store_season_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
                year=year,
            )
        case TrackStoreStep.COVER_SOURCE:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            await _show_store_sub_season_menu(
                message=message,
                state=state,
                settings=settings,
                buffer_version=_buffer_version_from_state(data),
                universe=universe,
                year=year,
                season=season,
            )
        case TrackStoreStep.ALBUM:
            selection = _selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            sub_season = _selected_sub_season(data)
            if sub_season is None:
                await handle_stale_selection(message=message, state=state)
                return
            buffered_messages = services.chat_message_buffer.peek_flat(message.chat.id)
            try:
                _ = validate_link_only_store_input(buffered_messages)
                await _show_store_cover_source_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    buffer_version=_buffer_version_from_state(data),
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                )
            except TrackInputError:
                await _show_store_sub_season_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    buffer_version=_buffer_version_from_state(data),
                    universe=universe,
                    year=year,
                    season=season,
                )
        case TrackStoreStep.FINAL:
            final_from_step = _final_from_step(data)
            if final_from_step is None:
                await handle_stale_selection(message=message, state=state)
                return
            if final_from_step is TrackStoreStep.SUB_SEASON:
                selection = _selected_universe_year_season(data)
                if selection is None:
                    await handle_stale_selection(message=message, state=state)
                    return
                universe, year, season = selection
                await _show_store_sub_season_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    buffer_version=_buffer_version_from_state(data),
                    universe=universe,
                    year=year,
                    season=season,
                )
                return
            if final_from_step is TrackStoreStep.COVER_SOURCE:
                selection = _selected_universe_year_season(data)
                sub_season = _selected_sub_season(data)
                if selection is None or sub_season is None:
                    await handle_stale_selection(message=message, state=state)
                    return
                universe, year, season = selection
                await _show_store_cover_source_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    buffer_version=_buffer_version_from_state(data),
                )
                return
            if final_from_step is TrackStoreStep.ALBUM:
                selection = _selected_universe_year_season(data)
                sub_season = _selected_sub_season(data)
                if selection is None or sub_season is None:
                    await handle_stale_selection(message=message, state=state)
                    return
                universe, year, season = selection
                try:
                    await _show_store_album_menu(
                        message=message,
                        state=state,
                        services=services,
                        settings=settings,
                        universe=universe,
                        year=year,
                        season=season,
                        sub_season=sub_season,
                        buffer_version=_buffer_version_from_state(data),
                    )
                except TrackInputError:
                    await handle_stale_selection(message=message, state=state)
                return
            await handle_stale_selection(message=message, state=state)


async def _execute_track_store(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    sub_season: SubSeason,
    universe: TrackUniverse,
    year: int,
    season: Season,
    completion_mode: TrackStoreCompletionMode = TrackStoreCompletionMode.SKIP,
) -> None:
    chat_id = message.chat.id
    expected_version = _buffer_version_from_state(await state.get_data())
    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    await message.edit_text(
        **selected_text(
            selected=_selected_store_path(
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
            ),
        ),
        reply_markup=None,
    )

    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    try:
        prepared_tracks = await prepare_tracks_from_buffer(
            bot=bot,
            messages=buffered_messages,
        )
    except TrackInputError as error:
        await state.clear()
        services.chat_message_buffer.flush(message.chat.id)
        await message.edit_text(str(error), reply_markup=None)
        return

    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    group = TrackGroup(universe=universe, year=year, season=season)
    # Multi-track store is best-effort convenience only.
    # Atomicity is per track, so partial success is allowed and reported.
    stored_titles: list[str] = []
    stored_track_infos: list[TrackInfo] = []
    try:
        for track in prepared_tracks:
            stored_info = await services.track_store.store(group, sub_season, track=track)
            stored_titles.append(track.title)
            stored_track_infos.append(stored_info)
    except Exception:
        logger.exception('track store failed')
        await state.clear()
        services.chat_message_buffer.flush(chat_id)
        if stored_titles:
            await message.answer(text='Storing failed\nStored titles:\n' + '\n'.join(stored_titles))
        else:
            await message.answer(text='Storing failed\nNo tracks were stored')
        return

    await state.clear()
    services.chat_message_buffer.flush(chat_id)
    await _complete_track_store(
        message=message,
        services=services,
        settings=settings,
        bot=bot,
        completion_mode=completion_mode,
        group=group,
        stored_track_infos=stored_track_infos,
    )


async def _on_store_sub_season_selected(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    sub_season: SubSeason,
    universe: TrackUniverse,
    year: int,
    season: Season,
) -> None:
    await _on_store_sub_season_selected_with_mode(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        sub_season=sub_season,
        universe=universe,
        year=year,
        season=season,
        completion_mode=None,
    )


async def _on_store_sub_season_selected_with_mode(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    sub_season: SubSeason,
    universe: TrackUniverse,
    year: int,
    season: Season,
    completion_mode: TrackStoreCompletionMode | None,
) -> None:
    chat_id = message.chat.id
    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    expected_version = _buffer_version_from_state(await state.get_data())
    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    try:
        parsed_cover_link_input = parse_cover_link_store_input(buffered_messages)
    except TrackInputError:
        parsed_cover_link_input = None
    if parsed_cover_link_input is not None:
        if completion_mode is None:
            await _show_store_final_menu(
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                from_step=TrackStoreStep.SUB_SEASON,
            )
            return
        await _execute_track_store_with_user_cover_link(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            universe=universe,
            year=year,
            season=season,
            sub_season=sub_season,
            parsed_cover_link_input=parsed_cover_link_input,
            completion_mode=completion_mode,
        )
        return

    if any(buffered_message.photo is not None for buffered_message in buffered_messages):
        if completion_mode is None:
            await _show_store_final_menu(
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                from_step=TrackStoreStep.SUB_SEASON,
            )
            return
        await _execute_track_store(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            sub_season=sub_season,
            universe=universe,
            year=year,
            season=season,
            completion_mode=completion_mode,
        )
        return

    try:
        is_link_only = False
        try:
            _ = validate_audio_only_store_input(buffered_messages)
        except TrackInputError:
            _ = validate_link_only_store_input(buffered_messages)
            is_link_only = True
        group = TrackGroup(universe=universe, year=year, season=season)
        try:
            tracks_by_sub_season = await services.track_store.list_tracks(group)
            sub_season_tracks = tracks_by_sub_season.get(sub_season, [])
            album_options = _unique_album_options(sub_season_tracks)
        except TrackGroupNotFoundError:
            album_options = []
        if is_link_only:
            if album_options:
                await _show_store_cover_source_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    buffer_version=expected_version,
                )
                return
            if completion_mode is None:
                await _show_store_final_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    from_step=TrackStoreStep.SUB_SEASON,
                )
                return
            await _execute_track_store_with_auto_cover(
                message=message,
                state=state,
                services=services,
                settings=settings,
                bot=bot,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                completion_mode=completion_mode,
            )
            return

        if album_options:
            await _show_store_album_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
                buffer_version=expected_version,
            )
            return
        raise TrackInputError('Invalid input')
    except TrackInputError, TrackGroupNotFoundError:
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
        )


async def _execute_track_store_with_auto_cover(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    completion_mode: TrackStoreCompletionMode = TrackStoreCompletionMode.SKIP,
) -> None:
    chat_id = message.chat.id
    expected_version = _buffer_version_from_state(await state.get_data())
    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    await message.edit_text(
        **selected_text(
            selected=[
                *_selected_store_path(
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                ),
                'Auto',
            ],
        ),
        reply_markup=None,
    )

    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    try:
        parsed_link_input = prepare_link_only_track_from_buffer(messages=buffered_messages)
        try:
            downloaded_result = await download_link_audio_and_cover(
                parsed_link_input.url,
                with_metadata=parsed_link_input.requires_metadata,
                max_duration=_derived_track_link_source_max_duration(settings),
            )
            artists, title = _resolve_track_link_metadata(
                parsed_link_input=parsed_link_input,
                metadata=downloaded_result.metadata,
            )
        except YtDlpMetadataError as error:
            if not _is_expected_track_metadata_unavailable(error):
                logger.exception('track link metadata unavailable')
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='No artists and title available',
            )
            return
        except TrackLinkDownloadError as error:
            if _is_youtube_cookies_required_error(error):
                await _invalidate_track_intake_buffer(
                    message=message,
                    state=state,
                    services=services,
                    text='Cookies required',
                )
                return
            logger.exception('track link download failed')
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Download failed',
            )
            return
        track = Track(
            artists=artists,
            title=title,
            audio=downloaded_result.audio,
            cover=downloaded_result.cover,
        )
        group = TrackGroup(universe=universe, year=year, season=season)
        stored_track_info = await services.track_store.store(group, sub_season, track=track)
    except TrackInputError, TrackGroupNotFoundError:
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
        )
        return
    except Exception:
        logger.exception('track store failed')
        await state.clear()
        services.chat_message_buffer.flush(chat_id)
        await message.answer(text='Storing failed\nNo tracks were stored')
        return

    await state.clear()
    services.chat_message_buffer.flush(chat_id)
    await _complete_track_store(
        message=message,
        services=services,
        settings=settings,
        bot=bot,
        completion_mode=completion_mode,
        group=group,
        stored_track_infos=[stored_track_info],
    )


async def _execute_track_store_with_user_cover_link(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    parsed_cover_link_input: CoverLinkTrackInput,
    completion_mode: TrackStoreCompletionMode = TrackStoreCompletionMode.SKIP,
) -> None:
    chat_id = message.chat.id
    expected_version = _buffer_version_from_state(await state.get_data())
    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    await message.edit_text(
        **selected_text(
            selected=_selected_store_path(
                universe=universe,
                year=year,
                season=season,
                sub_season=sub_season,
            ),
        ),
        reply_markup=None,
    )

    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    try:
        photo_message = next(
            (buffered_message for buffered_message in buffered_messages if buffered_message.photo), None
        )
        if photo_message is None or not photo_message.photo:
            raise TrackInputError('Invalid input')

        telegram_file = await bot.get_file(photo_message.photo[-1].file_id)
        if telegram_file.file_path is None:
            raise TrackInputError('Invalid input')
        downloaded_cover = await bot.download_file(telegram_file.file_path)
        if downloaded_cover is None:
            raise TrackInputError('Invalid input')
        cover = normalize_cover_to_jpg(downloaded_cover.read())

        try:
            downloaded_result = await download_audio_as_opus(
                parsed_cover_link_input.url,
                with_cover=False,
                with_metadata=parsed_cover_link_input.requires_metadata,
                max_duration=_derived_track_link_source_max_duration(settings),
            )
            link_only_input = LinkOnlyTrackInput(
                url=parsed_cover_link_input.url,
                artists=parsed_cover_link_input.artists,
                title=parsed_cover_link_input.title,
            )
            artists, title = _resolve_track_link_metadata(
                parsed_link_input=link_only_input,
                metadata=downloaded_result.metadata,
            )
        except YtDlpMetadataError as error:
            if not _is_expected_track_metadata_unavailable(error):
                logger.exception('track link metadata unavailable')
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='No artists and title available',
            )
            return
        except Exception as error:
            if _is_youtube_cookies_required_error(error):
                await _invalidate_track_intake_buffer(
                    message=message,
                    state=state,
                    services=services,
                    text='Cookies required',
                )
                return
            logger.exception('track link download failed')
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Download failed',
            )
            return

        group = TrackGroup(universe=universe, year=year, season=season)
        stored_track_info = await services.track_store.store(
            group,
            sub_season,
            track=Track(
                artists=artists,
                title=title,
                audio=FileBytes(data=downloaded_result.audio, extension=Extension.OPUS),
                cover=FileBytes(data=cover, extension=Extension.JPG),
                album_id=None,
            ),
        )
    except TrackInputError, TrackGroupNotFoundError:
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
        )
        return
    except Exception:
        logger.exception('track store failed')
        await state.clear()
        services.chat_message_buffer.flush(chat_id)
        await message.answer(text='Storing failed\nNo tracks were stored')
        return

    await state.clear()
    services.chat_message_buffer.flush(chat_id)
    await _complete_track_store(
        message=message,
        services=services,
        settings=settings,
        bot=bot,
        completion_mode=completion_mode,
        group=group,
        stored_track_infos=[stored_track_info],
    )


def _album_label_from_options(*, album_options: list[tuple[str, str]], album_id: str) -> str:
    for title, candidate_album_id in album_options:
        if candidate_album_id == album_id:
            return title
    return album_id


async def _resolve_album_label(
    *,
    services: Services,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    album_id: str,
) -> str:
    group = TrackGroup(universe=universe, year=year, season=season)
    try:
        tracks_by_sub_season = await services.track_store.list_tracks(group)
    except TrackGroupNotFoundError:
        return album_id
    sub_season_tracks = tracks_by_sub_season.get(sub_season, [])
    album_options = _unique_album_options(sub_season_tracks)
    return _album_label_from_options(album_options=album_options, album_id=album_id)


async def _execute_track_store_with_album_reuse(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
    album_id: str,
    completion_mode: TrackStoreCompletionMode = TrackStoreCompletionMode.SKIP,
) -> None:
    chat_id = message.chat.id
    expected_version = _buffer_version_from_state(await state.get_data())
    if services.chat_message_buffer.version(chat_id) != expected_version:
        await handle_stale_selection(message=message, state=state)
        return

    album_label = await _resolve_album_label(
        services=services,
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
        album_id=album_id,
    )
    await message.edit_text(
        **selected_text(
            selected=[
                *_selected_store_path(
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                ),
                'Album',
                album_label,
            ],
        ),
        reply_markup=None,
    )

    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    try:
        try:
            track = await prepare_audio_only_track_from_buffer(
                bot=bot,
                messages=buffered_messages,
                album_id=album_id,
            )
        except TrackInputError as error:
            parsed_link_input = prepare_link_only_track_from_buffer(messages=buffered_messages)
            if parsed_link_input.requires_metadata:
                raise TrackInputError('Invalid input') from error
            artists = parsed_link_input.artists
            title = parsed_link_input.title
            if artists is None or title is None:
                raise TrackInputError('Invalid input') from error
            try:
                downloaded_audio = await download_link_audio(
                    parsed_link_input.url,
                    max_duration=_derived_track_link_source_max_duration(settings),
                )
            except TrackLinkDownloadError as error:
                if _is_youtube_cookies_required_error(error):
                    await _invalidate_track_intake_buffer(
                        message=message,
                        state=state,
                        services=services,
                        text='Cookies required',
                    )
                    return
                logger.exception('track link audio download failed')
                await _invalidate_track_intake_buffer(
                    message=message,
                    state=state,
                    services=services,
                    text='Download failed',
                )
                return
            track = Track(
                artists=artists,
                title=title,
                audio=downloaded_audio,
                cover=None,
                album_id=album_id,
            )
        group = TrackGroup(universe=universe, year=year, season=season)
        stored_track_info = await services.track_store.store(group, sub_season, track=track)
    except TrackInputError, TrackGroupNotFoundError:
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
        )
        return
    except ValueError as error:
        if _is_missing_album_error(error):
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Invalid input',
            )
            return
        raise
    except Exception:
        logger.exception('track store failed')
        await state.clear()
        services.chat_message_buffer.flush(chat_id)
        await message.answer(text='Storing failed\nNo tracks were stored')
        return

    await state.clear()
    services.chat_message_buffer.flush(chat_id)
    await _complete_track_store(
        message=message,
        services=services,
        settings=settings,
        bot=bot,
        completion_mode=completion_mode,
        group=group,
        stored_track_infos=[stored_track_info],
    )


async def _complete_track_store(
    *,
    message: Message,
    services: Services,
    settings: Settings,
    bot: Bot,
    completion_mode: TrackStoreCompletionMode,
    group: TrackGroup,
    stored_track_infos: list[TrackInfo],
) -> None:
    if completion_mode is TrackStoreCompletionMode.SKIP:
        await message.answer(text='Done')
        return
    await send_fetched_tracks_by_ids(
        bot=bot,
        chat_id=message.chat.id,
        group=group,
        track_ids=[track_info.id for track_info in stored_track_infos],
        services=services,
        settings=settings,
    )


async def _execute_track_update(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    action: TrackIntakeAction,
    bot: Bot,
    expected_buffer_version: int,
) -> None:
    chat_id = message.chat.id
    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    owns_buffer = False
    try:
        photo_message, audio_message = extract_single_photo_audio_messages(buffered_messages)
        group, track_id = extract_track_identity_from_photo_message(photo_message)
        tracks_by_sub_season = await services.track_store.list_tracks(group)
        matched_sub_season = _resolve_track_sub_season(
            tracks_by_sub_season,
            track_id,
        )
        # Track id may not exist in current manifest (e.g. stale cover identity)
        # -> treat as user-level invalid input
        if matched_sub_season is None:
            raise TrackInputError('Invalid input')
        if services.chat_message_buffer.version(chat_id) != expected_buffer_version:
            await handle_stale_selection(message=message, state=state)
            return
        services.chat_message_buffer.flush(chat_id)
        await state.clear()
        owns_buffer = True
        action_label = 'Track' if action is TrackIntakeAction.TRACK else 'Instrumental'
        selected = [
            'Replace',
            action_label,
            _format_universe(group.universe),
            str(group.year),
            str(int(group.season)),
        ]
        if matched_sub_season.exists:
            selected.append(matched_sub_season.value)
        await message.edit_text(
            **selected_text(selected=selected),
            reply_markup=None,
        )
        prepared_audio = await prepare_audio_from_message(bot=bot, audio_message=audio_message)
        if action is TrackIntakeAction.TRACK:
            await services.track_store.update(group, track_id, track=prepared_audio)
        else:
            await services.track_store.update(group, track_id, instrumental=prepared_audio)
    except (
        TrackInputError,
        InvalidTrackIdentityError,
        TrackGroupNotFoundError,
        TrackInvalidAudioFormatError,
        InvalidExtensionError,
    ):
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
            flush_buffer=not owns_buffer,
        )
        return
    except ValueError as error:
        if _is_missing_track_error(error):
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Invalid input',
                flush_buffer=not owns_buffer,
            )
            return
        raise
    except (
        TrackUpdateManifestSyncError,
        TrackManifestCorruptedError,
        TrackPresetsCorruptedError,
    ):
        raise

    await message.answer('Done')


async def _execute_track_remove(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    action: TrackIntakeAction,
    expected_buffer_version: int,
) -> None:
    chat_id = message.chat.id
    buffered_messages = services.chat_message_buffer.peek_flat(chat_id)
    owns_buffer = False
    try:
        photo_messages = extract_photo_messages_for_remove(buffered_messages)
        targets = [extract_track_identity_from_photo_message(photo_message) for photo_message in photo_messages]
        if len(set(targets)) != len(targets):
            raise TrackInputError('Invalid input')
        tracks_by_group: dict[TrackGroup, dict[SubSeason, list[TrackInfo]]] = {}
        for group, track_id in targets:
            if group not in tracks_by_group:
                tracks_by_group[group] = await services.track_store.list_tracks(group)
            if _resolve_track_sub_season(tracks_by_group[group], track_id) is None:
                raise TrackInputError('Invalid input')
        if services.chat_message_buffer.version(chat_id) != expected_buffer_version:
            await handle_stale_selection(message=message, state=state)
            return

        services.chat_message_buffer.flush(chat_id)
        await state.clear()
        owns_buffer = True
        action_label = 'Track' if action is TrackIntakeAction.REMOVE_TRACK else 'Instrumental'
        await message.edit_text(
            **selected_text(selected=['Remove', action_label]),
            reply_markup=None,
        )

        for group, track_id in targets:
            if action is TrackIntakeAction.REMOVE_TRACK:
                await services.track_store.remove(group, track_id)
            else:
                await services.track_store.remove_instrumental(group, track_id)
    except (
        TrackInputError,
        InvalidTrackIdentityError,
        TrackGroupNotFoundError,
    ):
        await _invalidate_track_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
            flush_buffer=not owns_buffer,
        )
        return
    except ValueError as error:
        if _is_missing_track_error(error) or _is_missing_instrumental_error(error):
            await _invalidate_track_intake_buffer(
                message=message,
                state=state,
                services=services,
                text='Invalid input',
                flush_buffer=not owns_buffer,
            )
            return
        raise
    except (
        TrackRemoveManifestSyncError,
        TrackManifestCorruptedError,
    ):
        raise

    await message.answer('Done')


async def _invalidate_track_intake_buffer(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    text: str,
    flush_buffer: bool = True,
) -> None:
    await state.clear()
    if flush_buffer:
        services.chat_message_buffer.flush(message.chat.id)
    await message.edit_text(text, reply_markup=None)


def _derived_track_link_source_max_duration(settings: Settings) -> timedelta:
    """Return a coarse source cap from the final duration and slowest assumed variant speed.

    If presets later generate a speed below `slowest_variant_speed`, link-ingested
    source audio may be capped too short to produce a full-length variant.
    """
    return timedelta(
        seconds=math.ceil(
            settings.variant_max_duration.total_seconds() / settings.slowest_variant_speed,
        )
    )


async def _set_track_store_context(
    *,
    state: FSMContext,
    fsm_state: State,
    menu_message_id: int,
    buffer_version: int,
    universe: TrackUniverse | None = None,
    year: int | None = None,
    season: Season | None = None,
    sub_season: SubSeason | None = None,
) -> None:
    await state.set_state(fsm_state)
    await state.update_data(
        mode=_TRACK_STORE_MODE,
        menu_message_id=menu_message_id,
        buffer_version=buffer_version,
        universe=universe,
        year=year,
        season=season,
        sub_season=sub_season,
    )


async def _validate_track_store_callback(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    step: TrackStoreStep,
) -> bool:
    if not await validate_flow_state(
        message=message,
        state=state,
        expected_mode=_TRACK_STORE_MODE,
        expected_state=_state_for_step(step),
    ):
        return False

    data = await state.get_data()
    if _buffer_version_from_state(data) != services.chat_message_buffer.version(message.chat.id):
        await handle_stale_selection(message=message, state=state)
        return False
    return True


def _state_for_step(step: TrackStoreStep) -> State:
    match step:
        case TrackStoreStep.UNIVERSE:
            return TrackStoreFlow.universe
        case TrackStoreStep.YEAR:
            return TrackStoreFlow.year
        case TrackStoreStep.SEASON:
            return TrackStoreFlow.season
        case TrackStoreStep.SUB_SEASON:
            return TrackStoreFlow.sub_season
        case TrackStoreStep.COVER_SOURCE:
            return TrackStoreFlow.cover_source
        case TrackStoreStep.ALBUM:
            return TrackStoreFlow.album
        case TrackStoreStep.FINAL:
            return TrackStoreFlow.final


def _available_store_seasons(*, year: int, today: date) -> list[Season]:
    if year != today.year:
        return list(Season)
    max_season = Season.from_month(today.month)
    return [season for season in Season if season <= max_season]


def _buffer_version_from_state(data: dict[str, object]) -> int:
    buffer_version = data.get('buffer_version')
    if not isinstance(buffer_version, int):
        raise ValueError('Missing track buffer version')
    return buffer_version


def _selected_universe(data: dict[str, object]) -> TrackUniverse | None:
    universe = data.get('universe')
    if isinstance(universe, TrackUniverse):
        return universe
    return None


def _selected_universe_year(data: dict[str, object]) -> tuple[TrackUniverse, int] | None:
    universe = _selected_universe(data)
    year = data.get('year')
    if universe is None or not isinstance(year, int):
        return None
    return universe, year


def _selected_universe_year_season(data: dict[str, object]) -> tuple[TrackUniverse, int, Season] | None:
    selection = _selected_universe_year(data)
    season = data.get('season')
    if selection is None or not isinstance(season, Season):
        return None
    universe, year = selection
    return universe, year, season


def _selected_sub_season(data: dict[str, object]) -> SubSeason | None:
    sub_season = data.get('sub_season')
    if isinstance(sub_season, SubSeason):
        return sub_season
    return None


def _parse_track_store_completion_mode(value: str) -> TrackStoreCompletionMode | None:
    if value == _TRACK_STORE_FINAL_SKIP:
        return TrackStoreCompletionMode.SKIP
    if value == _TRACK_STORE_FINAL_FETCH:
        return TrackStoreCompletionMode.FETCH
    return None


def _final_from_step(data: dict[str, object]) -> TrackStoreStep | None:
    raw_value = data.get('final_from_step')
    if not isinstance(raw_value, str):
        return None
    try:
        final_from_step = TrackStoreStep(raw_value)
    except ValueError:
        return None
    if final_from_step not in (TrackStoreStep.SUB_SEASON, TrackStoreStep.COVER_SOURCE, TrackStoreStep.ALBUM):
        return None
    return final_from_step


def _final_album_id(data: dict[str, object]) -> str | None:
    value = data.get('final_album_id')
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _selected_store_path(
    *,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
) -> list[str]:
    selected = ['Store', _format_universe(universe), str(year), str(int(season))]
    if sub_season.exists:
        selected.append(_format_sub_season(sub_season))
    return selected


def _is_missing_track_error(error: ValueError) -> bool:
    return str(error).startswith('Track id ') and ' does not exist in group ' in str(error)


def _is_missing_instrumental_error(error: ValueError) -> bool:
    return str(error).startswith('Track id ') and ' does not have an instrumental in the provided group' in str(error)


def _is_missing_album_error(error: ValueError) -> bool:
    return str(error).startswith('Album id ') and ' does not exist in group ' in str(error)


def _resolve_track_link_metadata(
    *,
    parsed_link_input: LinkOnlyTrackInput,
    metadata: object,
) -> tuple[tuple[str, ...], str]:
    if not parsed_link_input.requires_metadata:
        artists = parsed_link_input.artists
        title = parsed_link_input.title
        if artists is None or title is None:
            raise TrackInputError('Invalid input')
        return artists, title

    if metadata is None:
        raise YtDlpMetadataError('yt-dlp did not produce metadata output')
    artists = getattr(metadata, 'artists', None)
    title = getattr(metadata, 'title', None)
    if not artists or not title:
        raise YtDlpMetadataError('yt-dlp produced incomplete metadata output')
    return tuple(artists), title


def _unique_album_options(tracks: list[TrackInfo]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for track in tracks:
        if track.album_id in seen:
            continue
        seen.add(track.album_id)
        options.append((track.title, track.album_id))
    return options


def _validate_store_input_at_entry(messages: list[Message]) -> None:
    has_video = any(
        message.video is not None or getattr(message, 'animation', None) is not None for message in messages
    )
    has_photo = any(message.photo is not None for message in messages)
    has_audio = any(extract_track_audio_attachment(message) is not None for message in messages)
    if has_video:
        raise TrackInputError('Invalid input')
    try:
        _ = parse_cover_link_store_input(messages)
        return
    except TrackInputError:
        pass
    if has_photo:
        validate_track_batch(extract_store_messages(messages))
        return
    if has_audio and any(_first_non_empty_line_is_supported_store_link(message.text) for message in messages):
        raise TrackInputError('Invalid input')
    try:
        validate_audio_only_store_input(messages)
    except TrackInputError:
        validate_link_only_store_input(messages)


def _first_non_empty_line_is_supported_store_link(text: str | None) -> bool:
    if text is None:
        return False
    for line in text.splitlines():
        normalized_line = line.strip()
        if not normalized_line:
            continue
        return is_supported_youtube_store_url(normalized_line)
    return False


def _album_selection_keyboard(*, album_options: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text=title,
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.SELECT,
                    step=TrackStoreStep.ALBUM,
                    value=album_id,
                ).pack(),
            )
        ]
        for title, album_id in album_options
    ]
    rows.reverse()
    rows.append(
        [
            back_button(
                callback_data=TrackStoreCallbackData(
                    action=TrackStoreAction.BACK,
                    step=TrackStoreStep.ALBUM,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _resolve_track_sub_season(
    tracks_by_sub_season: dict[SubSeason, list[TrackInfo]],
    track_id: str,
) -> SubSeason | None:
    for sub_season, tracks in tracks_by_sub_season.items():
        if any(track.id == track_id for track in tracks):
            return sub_season
    return None


def _format_universe(universe: TrackUniverse) -> str:
    return universe.value.title()


def _format_sub_season(sub_season: SubSeason) -> str:
    if sub_season is SubSeason.NONE:
        return 'None'
    return sub_season.value
