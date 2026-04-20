from collections.abc import Mapping, Sequence
from datetime import date
from enum import StrEnum, auto
from importlib.resources import files

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InputMediaAudio,
    Message,
)

from timeline_hub.handlers.menu import (
    back_button,
    callback_message,
    dummy_button,
    fixed_option_keyboard,
    handle_stale_selection,
    selected_text,
    selection_text,
    stacked_keyboard,
    validate_flow_state,
    width_reserved_text,
)
from timeline_hub.services.container import Services
from timeline_hub.services.track_store import (
    FetchedVariant,
    FetchedVariants,
    Season,
    SubSeason,
    TrackGroup,
    TrackGroupNotFoundError,
    TrackInfo,
    TrackStore,
    TrackUniverse,
)
from timeline_hub.settings import Settings
from timeline_hub.types import Extension

router = Router()
_TRACK_GET_MODE = 'track_get'
_TRACK_BACK_VALUE = 'back'
_AUDIO_THUMBNAIL_NAME = 'track-thumbnail.jpg'
_AUDIO_THUMBNAIL_BYTES = files('timeline_hub').joinpath(f'assets/{_AUDIO_THUMBNAIL_NAME}').read_bytes()


class RetrieveEntryAction(StrEnum):
    GET = auto()
    CANCEL = auto()


class RetrieveEntryCallbackData(CallbackData, prefix='track_retrieve_entry'):
    action: RetrieveEntryAction


class TrackRetrieveAction(StrEnum):
    SELECT = auto()
    BACK = auto()


class TrackRetrieveStep(StrEnum):
    UNIVERSE = auto()
    YEAR = auto()
    SEASON = auto()
    SUB_SEASON = auto()


class TrackRetrieveCallbackData(CallbackData, prefix='track_retrieve'):
    action: TrackRetrieveAction
    step: TrackRetrieveStep
    value: str


class TrackRetrieveFlow(StatesGroup):
    universe = State()
    year = State()
    season = State()
    sub_season = State()


@router.message(F.text == 'Tracks')
async def on_tracks(message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.answer(
        **width_reserved_text(
            text='Select action:',
            message_width=settings.message_width,
        ),
        reply_markup=_track_entry_reply_markup(),
    )


@router.callback_query(
    RetrieveEntryCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_retrieve_entry(
    callback: CallbackQuery,
    callback_data: RetrieveEntryCallbackData,
    services: Services,
    settings: Settings,
    state: FSMContext,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.action is RetrieveEntryAction.CANCEL:
        await state.clear()
        await message.edit_text(
            **selected_text(selected='Cancel'),
            reply_markup=None,
        )
        return

    groups = await services.track_store.list_groups()
    await _show_retrieve_universe_menu(
        message=message,
        state=state,
        settings=settings,
        groups=groups,
    )


@router.callback_query(
    TrackRetrieveCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_retrieve_menu(
    callback: CallbackQuery,
    callback_data: TrackRetrieveCallbackData,
    bot: Bot,
    services: Services,
    settings: Settings,
    state: FSMContext,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if not await _validate_track_retrieve_callback(
        message=message,
        state=state,
        step=callback_data.step,
    ):
        return

    if callback_data.action is TrackRetrieveAction.BACK:
        await _on_retrieve_back(
            message=message,
            state=state,
            settings=settings,
            step=callback_data.step,
        )
        return

    await _on_retrieve_select(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        callback_data=callback_data,
    )


async def _on_retrieve_back(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    step: TrackRetrieveStep,
) -> None:
    data = await state.get_data()

    match step:
        case TrackRetrieveStep.UNIVERSE:
            await _show_retrieve_entry_menu(
                message=message,
                state=state,
                settings=settings,
            )
        case TrackRetrieveStep.YEAR:
            if not await _show_retrieve_universe_menu(
                message=message,
                state=state,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)
        case TrackRetrieveStep.SEASON:
            universe = _selected_universe(data)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_retrieve_year_menu(
                message=message,
                state=state,
                universe=universe,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)
        case TrackRetrieveStep.SUB_SEASON:
            selection = _selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            if not await _show_retrieve_season_menu(
                message=message,
                state=state,
                universe=universe,
                year=year,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)


async def _on_retrieve_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    callback_data: TrackRetrieveCallbackData,
) -> None:
    data = await state.get_data()

    match callback_data.step:
        case TrackRetrieveStep.UNIVERSE:
            universe = _parse_universe(callback_data.value)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_retrieve_year_menu(
                message=message,
                state=state,
                universe=universe,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case TrackRetrieveStep.YEAR:
            universe = _selected_universe(data)
            year = _parse_year(callback_data.value)
            if universe is None or year is None:
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_retrieve_season_menu(
                message=message,
                state=state,
                universe=universe,
                year=year,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case TrackRetrieveStep.SEASON:
            selection = _selected_universe_year(data)
            season = _parse_season(callback_data.value)
            if selection is None or season is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            group = TrackGroup(universe=universe, year=year, season=season)

            try:
                tracks_by_sub_season = await services.track_store.list_tracks(group)
            except TrackGroupNotFoundError:
                await handle_stale_selection(message=message, state=state)
                return

            if not await _show_retrieve_sub_season_menu(
                message=message,
                state=state,
                group=group,
                tracks_by_sub_season=tracks_by_sub_season,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case TrackRetrieveStep.SUB_SEASON:
            selection = _selected_universe_year_season(data)
            sub_season = _parse_sub_season(callback_data.value)
            if selection is None or sub_season is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            await _execute_track_get(
                message=message,
                state=state,
                services=services,
                bot=bot,
                group=TrackGroup(universe=universe, year=year, season=season),
                sub_season=sub_season,
            )


async def _show_retrieve_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    groups: list[TrackGroup] | None = None,
) -> bool:
    if groups is None:
        groups = _groups_from_data(await state.get_data())
    if groups is None:
        return False

    universes = _available_universes(groups)
    await _set_track_retrieve_context(
        state=state,
        fsm_state=TrackRetrieveFlow.universe,
        menu_message_id=message.message_id,
        groups=groups,
    )
    await message.edit_text(
        **selection_text(
            selected=['Get'],
            prompt='Select universe:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(TrackUniverse),
            available_options=universes,
            build_button=lambda universe: InlineKeyboardButton(
                text=_format_universe(universe),
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.SELECT,
                    step=TrackRetrieveStep.UNIVERSE,
                    value=universe.value,
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.BACK,
                    step=TrackRetrieveStep.UNIVERSE,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )
    return True


async def _show_retrieve_year_menu(
    *,
    message: Message,
    state: FSMContext,
    universe: TrackUniverse,
    settings: Settings,
) -> bool:
    data = await state.get_data()
    groups = _groups_from_data(data)
    if groups is None:
        return False

    years = _available_years(groups, universe=universe)
    if not years:
        return False
    year_options = list(range(date.today().year, settings.min_clip_year - 1, -1))

    await _set_track_retrieve_context(
        state=state,
        fsm_state=TrackRetrieveFlow.year,
        menu_message_id=message.message_id,
        groups=groups,
        universe=universe,
    )
    await message.edit_text(
        **selection_text(
            selected=['Get', _format_universe(universe)],
            prompt='Select year:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=year_options,
            available_options=years,
            build_button=lambda year: InlineKeyboardButton(
                text=str(year),
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.SELECT,
                    step=TrackRetrieveStep.YEAR,
                    value=str(year),
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.BACK,
                    step=TrackRetrieveStep.YEAR,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )
    return True


async def _show_retrieve_season_menu(
    *,
    message: Message,
    state: FSMContext,
    universe: TrackUniverse,
    year: int,
    settings: Settings,
) -> bool:
    data = await state.get_data()
    groups = _groups_from_data(data)
    if groups is None:
        return False

    seasons = _available_seasons(groups, universe=universe, year=year)
    if not seasons:
        return False

    await _set_track_retrieve_context(
        state=state,
        fsm_state=TrackRetrieveFlow.season,
        menu_message_id=message.message_id,
        groups=groups,
        universe=universe,
        year=year,
    )
    await message.edit_text(
        **selection_text(
            selected=['Get', _format_universe(universe), str(year)],
            prompt='Select season:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(Season),
            available_options=seasons,
            build_button=lambda season: InlineKeyboardButton(
                text=str(int(season)),
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.SELECT,
                    step=TrackRetrieveStep.SEASON,
                    value=str(int(season)),
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.BACK,
                    step=TrackRetrieveStep.SEASON,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )
    return True


async def _show_retrieve_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    group: TrackGroup,
    tracks_by_sub_season: Mapping[SubSeason, list[TrackInfo]],
    settings: Settings,
) -> bool:
    sub_seasons = _available_sub_seasons(tracks_by_sub_season)
    if not sub_seasons:
        return False

    await _set_track_retrieve_context(
        state=state,
        fsm_state=TrackRetrieveFlow.sub_season,
        menu_message_id=message.message_id,
        universe=group.universe,
        year=group.year,
        season=group.season,
        tracks_by_sub_season=tracks_by_sub_season,
    )
    await message.edit_text(
        **selection_text(
            selected=['Get', _format_universe(group.universe), str(group.year), str(int(group.season))],
            prompt='Select sub-season:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(SubSeason),
            available_options=sub_seasons,
            build_button=lambda sub_season: InlineKeyboardButton(
                text=_format_sub_season(sub_season),
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.SELECT,
                    step=TrackRetrieveStep.SUB_SEASON,
                    value=sub_season.value,
                ).pack(),
            ),
            back_button=back_button(
                callback_data=TrackRetrieveCallbackData(
                    action=TrackRetrieveAction.BACK,
                    step=TrackRetrieveStep.SUB_SEASON,
                    value=_TRACK_BACK_VALUE,
                ).pack(),
            ),
        ),
    )
    return True


async def _execute_track_get(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    bot: Bot,
    group: TrackGroup,
    sub_season: SubSeason,
) -> None:
    data = await state.get_data()
    tracks_by_sub_season = _tracks_by_sub_season_from_data(data)
    if tracks_by_sub_season is None:
        await handle_stale_selection(message=message, state=state)
        return

    track_infos = tracks_by_sub_season.get(sub_season)
    if track_infos is None or not track_infos:
        await handle_stale_selection(message=message, state=state)
        return

    await message.edit_text(
        **selected_text(
            selected=_selected_retrieve_path(
                universe=group.universe,
                year=group.year,
                season=group.season,
                sub_season=sub_season,
            )
        ),
        reply_markup=None,
    )
    await state.clear()

    fetched_tracks: list[FetchedVariants] = []
    try:
        for track_info in track_infos:
            try:
                fetched_tracks.append(await services.track_store.fetch(group, track_info.id))
            except ValueError as error:
                missing_track_error = (
                    f'Track id {track_info.id} does not exist in group '
                    f'{group.universe.value}-{group.year}-{int(group.season)}'
                )
                if str(error) != missing_track_error:
                    raise
                await handle_stale_selection(message=message, state=state)
                return
    except TrackGroupNotFoundError:
        await handle_stale_selection(message=message, state=state)
        return

    for fetched_track in reversed(fetched_tracks):
        await _send_fetched_track(
            bot=bot,
            chat_id=message.chat.id,
            group=group,
            fetched_track=fetched_track,
        )


async def _send_fetched_track(
    *,
    bot: Bot,
    chat_id: int,
    group: TrackGroup,
    fetched_track: FetchedVariants,
) -> None:
    _validate_variant_count(fetched_track.variants)
    if fetched_track.instrumental_variants is not None:
        _validate_variant_count(fetched_track.instrumental_variants)

    await bot.send_photo(
        chat_id=chat_id,
        photo=BufferedInputFile(
            fetched_track.cover.data,
            filename=_cover_filename(group=group, track_id=fetched_track.track_id),
        ),
        caption=fetched_track.title,
    )
    await _send_variant_audio(
        bot=bot,
        chat_id=chat_id,
        variants=fetched_track.variants,
    )
    if fetched_track.instrumental_variants is not None:
        await _send_variant_audio(
            bot=bot,
            chat_id=chat_id,
            variants=fetched_track.instrumental_variants,
        )


async def _send_variant_audio(
    *,
    bot: Bot,
    chat_id: int,
    variants: Sequence[FetchedVariant],
) -> None:
    _validate_variant_count(variants)

    if len(variants) == 1:
        await bot.send_audio(
            chat_id=chat_id,
            audio=BufferedInputFile(
                variants[0].audio.data,
                filename=_variant_filename(variants[0], index=1),
            ),
            thumbnail=_audio_thumbnail(),
            performer='\u2009',
            title=_variant_title(variants[0]),
        )
        return

    await bot.send_media_group(
        chat_id=chat_id,
        media=[
            InputMediaAudio(
                media=BufferedInputFile(
                    variant.audio.data,
                    filename=_variant_filename(variant, index=index),
                ),
                thumbnail=_audio_thumbnail(),
                performer='\u2009',
                title=_variant_title(variant),
            )
            for index, variant in enumerate(variants, start=1)
        ],
    )


async def _show_retrieve_entry_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
) -> None:
    await state.clear()
    await message.edit_text(
        **width_reserved_text(
            text='Select action:',
            message_width=settings.message_width,
        ),
        reply_markup=_track_entry_reply_markup(),
    )


async def _set_track_retrieve_context(
    *,
    state: FSMContext,
    fsm_state: State,
    menu_message_id: int,
    groups: list[TrackGroup] | None = None,
    tracks_by_sub_season: Mapping[SubSeason, list[TrackInfo]] | None = None,
    universe: TrackUniverse | None = None,
    year: int | None = None,
    season: Season | None = None,
) -> None:
    existing_data = await state.get_data()
    if groups is None:
        groups = _groups_from_data(existing_data)
    if tracks_by_sub_season is None:
        tracks_by_sub_season = _tracks_by_sub_season_from_data(existing_data)

    await state.clear()
    await state.set_state(fsm_state)

    data: dict[str, object] = {
        'mode': _TRACK_GET_MODE,
        'menu_message_id': menu_message_id,
    }
    if groups is not None:
        data['groups'] = groups
    if tracks_by_sub_season is not None:
        data['tracks_by_sub_season'] = dict(tracks_by_sub_season)
    if universe is not None:
        data['universe'] = universe
    if year is not None:
        data['year'] = year
    if season is not None:
        data['season'] = season

    await state.update_data(data)


async def _validate_track_retrieve_callback(
    *,
    message: Message,
    state: FSMContext,
    step: TrackRetrieveStep,
) -> bool:
    return await validate_flow_state(
        message=message,
        state=state,
        expected_mode=_TRACK_GET_MODE,
        expected_state=_state_for_step(step),
    )


def _track_entry_reply_markup():
    return stacked_keyboard(
        buttons=[
            InlineKeyboardButton(
                text='Get',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.GET).pack(),
            ),
            dummy_button(),
            InlineKeyboardButton(
                text='Cancel',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.CANCEL).pack(),
            ),
        ]
    )


def _state_for_step(step: TrackRetrieveStep) -> State:
    match step:
        case TrackRetrieveStep.UNIVERSE:
            return TrackRetrieveFlow.universe
        case TrackRetrieveStep.YEAR:
            return TrackRetrieveFlow.year
        case TrackRetrieveStep.SEASON:
            return TrackRetrieveFlow.season
        case TrackRetrieveStep.SUB_SEASON:
            return TrackRetrieveFlow.sub_season


def _groups_from_data(data: Mapping[str, object]) -> list[TrackGroup] | None:
    groups = data.get('groups')
    if isinstance(groups, list) and all(isinstance(group, TrackGroup) for group in groups):
        return groups
    return None


def _tracks_by_sub_season_from_data(data: Mapping[str, object]) -> dict[SubSeason, list[TrackInfo]] | None:
    raw_value = data.get('tracks_by_sub_season')
    if not isinstance(raw_value, dict):
        return None

    tracks_by_sub_season: dict[SubSeason, list[TrackInfo]] = {}
    for sub_season, track_infos in raw_value.items():
        if not isinstance(sub_season, SubSeason):
            return None
        if not isinstance(track_infos, list) or any(
            not isinstance(track_info, TrackInfo) for track_info in track_infos
        ):
            return None
        tracks_by_sub_season[sub_season] = track_infos
    return tracks_by_sub_season


def _selected_universe(data: Mapping[str, object]) -> TrackUniverse | None:
    universe = data.get('universe')
    if isinstance(universe, TrackUniverse):
        return universe
    return None


def _selected_universe_year(data: Mapping[str, object]) -> tuple[TrackUniverse, int] | None:
    universe = _selected_universe(data)
    year = data.get('year')
    if universe is None or not isinstance(year, int):
        return None
    return universe, year


def _selected_universe_year_season(data: Mapping[str, object]) -> tuple[TrackUniverse, int, Season] | None:
    selection = _selected_universe_year(data)
    season = data.get('season')
    if selection is None or not isinstance(season, Season):
        return None
    universe, year = selection
    return universe, year, season


def _available_universes(groups: Sequence[TrackGroup]) -> list[TrackUniverse]:
    universes: list[TrackUniverse] = []
    for group in groups:
        if group.universe not in universes:
            universes.append(group.universe)
    return universes


def _available_years(groups: Sequence[TrackGroup], *, universe: TrackUniverse) -> list[int]:
    years: list[int] = []
    for group in groups:
        if group.universe is universe and group.year not in years:
            years.append(group.year)
    return years


def _available_seasons(
    groups: Sequence[TrackGroup],
    *,
    universe: TrackUniverse,
    year: int,
) -> list[Season]:
    seasons: list[Season] = []
    for group in groups:
        if group.universe is universe and group.year == year and group.season not in seasons:
            seasons.append(group.season)
    return seasons


def _available_sub_seasons(tracks_by_sub_season: Mapping[SubSeason, list[TrackInfo]]) -> list[SubSeason]:
    return list(tracks_by_sub_season)


def _selected_retrieve_path(
    *,
    universe: TrackUniverse,
    year: int,
    season: Season,
    sub_season: SubSeason,
) -> list[str]:
    selected = ['Get', _format_universe(universe), str(year), str(int(season))]
    if sub_season is not SubSeason.NONE:
        selected.append(_format_sub_season(sub_season))
    return selected


def _parse_universe(value: str) -> TrackUniverse | None:
    try:
        return TrackUniverse(value)
    except ValueError:
        return None


def _parse_year(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _parse_season(value: str) -> Season | None:
    try:
        return Season(int(value))
    except ValueError:
        return None


def _parse_sub_season(value: str) -> SubSeason | None:
    try:
        return SubSeason(value)
    except ValueError:
        return None


def _format_universe(universe: TrackUniverse) -> str:
    return universe.value.title()


def _format_sub_season(sub_season: SubSeason) -> str:
    if sub_season is SubSeason.NONE:
        return 'None'
    return sub_season.value


def _cover_filename(*, group: TrackGroup, track_id: str) -> str:
    return f'{TrackStore.track_identity_to_string(group, track_id)}-cover{Extension.JPG.suffix}'


def _variant_title(variant: FetchedVariant) -> str:
    if variant.speed < 1.0:
        arrow = '⏪'
    elif variant.speed > 1.0:
        arrow = '⏩'
    else:
        raise ValueError('Fetched track variant speed must not be 1.0')

    if variant.level < 1:
        raise ValueError('Fetched track variant level must be >= 1')
    return arrow * variant.level


def _variant_filename(variant: FetchedVariant, *, index: int) -> str:
    return f'{index}{variant.audio.extension.suffix}'


def _audio_thumbnail() -> BufferedInputFile:
    return BufferedInputFile(_AUDIO_THUMBNAIL_BYTES, filename=_AUDIO_THUMBNAIL_NAME)


def _validate_variant_count(variants: Sequence[FetchedVariant]) -> None:
    if not variants:
        raise ValueError('Fetched track variants must not be empty')
    if len(variants) > 10:
        raise ValueError('Fetched track variants must contain at most 10 items')
