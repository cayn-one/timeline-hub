from datetime import date
from collections.abc import Sequence
from enum import StrEnum, auto

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InputMediaVideo, Message

from general_bot.services.clip_store import (
    Clip,
    ClipGroup,
    ClipGroupNotFoundError,
    ClipSubGroup,
    Scope,
    Season,
    SubSeason,
    Universe,
)
from general_bot.handlers.clips_common import (
    ALL_SCOPES_CALLBACK_VALUE,
    BACK_CALLBACK_VALUE,
    FETCH_STATE_BY_STEP,
    FLOW_FETCH,
    UNSET,
    FetchClipFlow,
    MenuAction,
    MenuStep,
    back_button,
    callback_message,
    encode_sub_season,
    fixed_option_keyboard,
    format_selection_value,
    handle_stale_selection,
    dummy_button,
    parse_scope,
    parse_season,
    parse_sub_season,
    parse_universe,
    parse_year,
    selected_text,
    selection_labels,
    selection_text,
    set_flow_context,
    single_button_keyboard,
    stacked_keyboard,
    terminate_menu,
    validate_flow_state,
    width_reserved_text,
)
from general_bot.settings import Settings
from general_bot.services.container import Services
from general_bot.types import ChatId

router = Router()


class FetchEntryAction(StrEnum):
    OPEN = auto()
    CANCEL = auto()


class FetchEntryCallbackData(CallbackData, prefix='clip_fetch_entry'):
    action: FetchEntryAction


class FetchCallbackData(CallbackData, prefix='clip_fetch'):
    action: MenuAction
    step: MenuStep
    value: str


@router.message(F.text == 'Clips')
async def on_clips(message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.answer(
        **width_reserved_text(
            text='Select action:',
            message_width=settings.message_width,
        ),
        reply_markup=_fetch_entry_reply_markup(),
    )


@router.callback_query(
    FetchEntryCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_fetch_entry(
    callback: CallbackQuery,
    callback_data: FetchEntryCallbackData,
    services: Services,
    settings: Settings,
    state: FSMContext,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.action is FetchEntryAction.CANCEL:
        await state.clear()
        await message.edit_text(
            **selected_text(selected='Cancel'),
            reply_markup=None,
        )
        return

    groups = await services.clip_store.list_groups()
    if not groups:
        await terminate_menu(
            message=message,
            state=state,
            text='No clips stored',
        )
        return

    await _show_fetch_year_menu(
        message=message,
        state=state,
        settings=settings,
        groups=groups,
    )


@router.callback_query(
    FetchCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_fetch_menu(
    callback: CallbackQuery,
    callback_data: FetchCallbackData,
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

    if not await validate_flow_state(
        message=message,
        state=state,
        expected_mode=FLOW_FETCH,
        expected_state=FETCH_STATE_BY_STEP[callback_data.step],
    ):
        return

    if callback_data.action is MenuAction.BACK:
        await _on_fetch_back(
            message=message,
            state=state,
            services=services,
            settings=settings,
            step=callback_data.step,
        )
        return

    await _on_fetch_select(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        callback_data=callback_data,
    )


async def _on_fetch_back(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    step: MenuStep,
) -> None:
    data = await state.get_data()

    match step:
        case MenuStep.YEAR:
            await _show_fetch_entry_menu(message=message, state=state, settings=settings)

        case MenuStep.SEASON:
            if not await _show_fetch_year_menu(
                message=message,
                state=state,
                settings=settings,
                services=services,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.UNIVERSE:
            year = data.get('year')
            if not isinstance(year, int):
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_fetch_season_menu(
                message=message,
                state=state,
                year=year,
                services=services,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SUB_SEASON:
            year = data.get('year')
            season = data.get('season')
            if not isinstance(year, int) or not isinstance(season, Season):
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_fetch_universe_menu(
                message=message,
                state=state,
                year=year,
                season=season,
                services=services,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SCOPE:
            year = data.get('year')
            season = data.get('season')
            universe = data.get('universe')
            if not isinstance(year, int) or not isinstance(season, Season) or not isinstance(universe, Universe):
                await handle_stale_selection(message=message, state=state)
                return

            sub_groups = await _fetch_sub_groups(
                services=services,
                year=year,
                season=season,
                universe=universe,
            )
            if sub_groups is None:
                await handle_stale_selection(message=message, state=state)
                return

            if _fetch_sub_season_options(sub_groups) == [SubSeason.NONE]:
                if not await _show_fetch_universe_menu(
                    message=message,
                    state=state,
                    year=year,
                    season=season,
                    services=services,
                    settings=settings,
                ):
                    await handle_stale_selection(message=message, state=state)
                return

            if not await _show_fetch_sub_season_menu(
                message=message,
                state=state,
                year=year,
                season=season,
                universe=universe,
                services=services,
                settings=settings,
                sub_groups=sub_groups,
            ):
                await handle_stale_selection(message=message, state=state)


async def _on_fetch_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    callback_data: FetchCallbackData,
) -> None:
    data = await state.get_data()

    match callback_data.step:
        case MenuStep.YEAR:
            year = parse_year(callback_data.value)
            if year is None or not await _show_fetch_season_menu(
                message=message,
                state=state,
                year=year,
                services=services,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SEASON:
            year = data.get('year')
            season = parse_season(callback_data.value)
            if not isinstance(year, int) or season is None or not await _show_fetch_universe_menu(
                message=message,
                state=state,
                year=year,
                season=season,
                services=services,
                settings=settings,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.UNIVERSE:
            year = data.get('year')
            season = data.get('season')
            universe = parse_universe(callback_data.value)
            if (
                not isinstance(year, int)
                or not isinstance(season, Season)
                or universe is None
                or not await _show_fetch_sub_season_menu(
                    message=message,
                    state=state,
                    year=year,
                    season=season,
                    universe=universe,
                    services=services,
                    settings=settings,
                )
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SUB_SEASON:
            year = data.get('year')
            season = data.get('season')
            universe = data.get('universe')
            sub_season = parse_sub_season(callback_data.value)
            if (
                not isinstance(year, int)
                or not isinstance(season, Season)
                or not isinstance(universe, Universe)
                or sub_season is UNSET
                or not await _show_fetch_scope_menu(
                    message=message,
                    state=state,
                    year=year,
                    season=season,
                    universe=universe,
                    sub_season=sub_season,
                    services=services,
                    settings=settings,
                )
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SCOPE:
            year = data.get('year')
            season = data.get('season')
            universe = data.get('universe')
            sub_season = data.get('sub_season', UNSET)
            if (
                not isinstance(year, int)
                or not isinstance(season, Season)
                or not isinstance(universe, Universe)
                or sub_season is UNSET
            ):
                await handle_stale_selection(message=message, state=state)
                return

            sub_groups = await _fetch_sub_groups(
                services=services,
                year=year,
                season=season,
                universe=universe,
            )
            if sub_groups is None:
                await handle_stale_selection(message=message, state=state)
                return

            scopes = _fetch_scope_options(sub_groups, sub_season)
            if not scopes:
                await handle_stale_selection(message=message, state=state)
                return

            if callback_data.value == ALL_SCOPES_CALLBACK_VALUE:
                selected_labels = _fetch_selection_labels(
                    year=year,
                    season=season,
                    universe=universe,
                    sub_season=sub_season,
                    scope='All',
                )
            else:
                scope = parse_scope(callback_data.value)
                if scope is None or scope not in scopes:
                    await handle_stale_selection(message=message, state=state)
                    return
                scopes = [scope]
                selected_labels = _fetch_selection_labels(
                    year=year,
                    season=season,
                    universe=universe,
                    sub_season=sub_season,
                    scope=scope,
                )

            await message.edit_text(
                **selection_text(selected=selected_labels),
                reply_markup=None,
            )
            try:
                await _send_fetch_scopes(
                    bot=bot,
                    chat_id=message.chat.id,
                    services=services,
                    year=year,
                    season=season,
                    universe=universe,
                    sub_season=sub_season,
                    scopes=scopes,
                )
            except ClipGroupNotFoundError:
                await handle_stale_selection(message=message, state=state)
                return
            await state.clear()


async def _show_fetch_year_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    services: Services | None = None,
    groups: list[ClipGroup] | None = None,
) -> bool:
    if groups is None:
        if services is None:
            return False
        groups = await services.clip_store.list_groups()

    available_years = _fetch_year_options(groups)
    if not available_years:
        return False
    year_universe = _fetch_year_universe(settings)

    await set_flow_context(
        state=state,
        mode=FLOW_FETCH,
        menu_message_id=message.message_id,
        fsm_state=FetchClipFlow.year,
    )
    await message.edit_text(
        **selection_text(
            selected=_fetch_selection_labels(),
            prompt='Select year:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=list(reversed(year_universe)),
            available_options=available_years,
            build_button=lambda year: _fetch_menu_button(
                step=MenuStep.YEAR,
                value=str(year),
                text=str(year),
            ),
            back_button=_fetch_back_button(step=MenuStep.YEAR),
        ),
    )
    return True


async def _show_fetch_season_menu(
    *,
    message: Message,
    state: FSMContext,
    year: int,
    services: Services,
    settings: Settings,
) -> bool:
    groups = await services.clip_store.list_groups()
    available_seasons = _fetch_season_options(groups, year=year)
    store_allowed_seasons = _fetch_store_allowed_seasons(year)
    available_seasons = [season for season in available_seasons if season in store_allowed_seasons]
    if not available_seasons:
        return False
    season_universe = _fetch_season_universe(year)

    await set_flow_context(
        state=state,
        mode=FLOW_FETCH,
        menu_message_id=message.message_id,
        fsm_state=FetchClipFlow.season,
        year=year,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select season:',
            selected=_fetch_selection_labels(year=year),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=season_universe,
            available_options=available_seasons,
            build_button=lambda season: _fetch_menu_button(
                step=MenuStep.SEASON,
                value=str(int(season)),
                text=str(int(season)),
            ),
            back_button=_fetch_back_button(step=MenuStep.SEASON),
        ),
    )
    return True


async def _show_fetch_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    year: int,
    season: Season,
    services: Services,
    settings: Settings,
) -> bool:
    groups = await services.clip_store.list_groups()
    available_universes = _fetch_universe_options(groups, year=year, season=season)
    if not available_universes:
        return False

    await set_flow_context(
        state=state,
        mode=FLOW_FETCH,
        menu_message_id=message.message_id,
        fsm_state=FetchClipFlow.universe,
        year=year,
        season=season,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select universe:',
            selected=_fetch_selection_labels(year=year, season=season),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(Universe),
            available_options=available_universes,
            build_button=lambda universe: _fetch_menu_button(
                step=MenuStep.UNIVERSE,
                value=universe.value,
                text=format_selection_value(universe),
            ),
            back_button=_fetch_back_button(step=MenuStep.UNIVERSE),
        ),
    )
    return True


async def _show_fetch_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    year: int,
    season: Season,
    universe: Universe,
    services: Services,
    settings: Settings,
    sub_groups: list[ClipSubGroup] | None = None,
) -> bool:
    if sub_groups is None:
        sub_groups = await _fetch_sub_groups(
            services=services,
            year=year,
            season=season,
            universe=universe,
        )
    if sub_groups is None:
        return False

    sub_seasons = _fetch_sub_season_options(sub_groups)
    if sub_seasons == [SubSeason.NONE]:
        return await _show_fetch_scope_menu(
            message=message,
            state=state,
            year=year,
            season=season,
            universe=universe,
            sub_season=SubSeason.NONE,
            services=services,
            settings=settings,
            sub_groups=sub_groups,
        )
    await set_flow_context(
        state=state,
        mode=FLOW_FETCH,
        menu_message_id=message.message_id,
        fsm_state=FetchClipFlow.sub_season,
        year=year,
        season=season,
        universe=universe,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select sub-season:',
            selected=_fetch_selection_labels(year=year, season=season, universe=universe),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(SubSeason),
            available_options=sub_seasons,
            build_button=lambda sub_season: _fetch_menu_button(
                step=MenuStep.SUB_SEASON,
                value=encode_sub_season(sub_season),
                text=format_selection_value(sub_season),
            ),
            back_button=_fetch_back_button(step=MenuStep.SUB_SEASON),
        ),
    )
    return True


async def _show_fetch_scope_menu(
    *,
    message: Message,
    state: FSMContext,
    year: int,
    season: Season,
    universe: Universe,
    sub_season: SubSeason,
    services: Services,
    settings: Settings,
    sub_groups: list[ClipSubGroup] | None = None,
) -> bool:
    if sub_groups is None:
        sub_groups = await _fetch_sub_groups(
            services=services,
            year=year,
            season=season,
            universe=universe,
        )
    if sub_groups is None:
        return False

    scopes = _fetch_scope_options(sub_groups, sub_season)
    if not scopes:
        return False

    await set_flow_context(
        state=state,
        mode=FLOW_FETCH,
        menu_message_id=message.message_id,
        fsm_state=FetchClipFlow.scope,
        year=year,
        season=season,
        universe=universe,
        sub_season=sub_season,
    )
    available_scope_options: list[Scope | str] = [ALL_SCOPES_CALLBACK_VALUE, *scopes]

    await message.edit_text(
        **selection_text(
            prompt='Select scope:',
            selected=_fetch_selection_labels(
                year=year,
                season=season,
                universe=universe,
                sub_season=sub_season,
            ),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=(ALL_SCOPES_CALLBACK_VALUE, *Scope),
            available_options=available_scope_options,
            build_button=lambda option: _fetch_menu_button(
                step=MenuStep.SCOPE,
                value=ALL_SCOPES_CALLBACK_VALUE if option == ALL_SCOPES_CALLBACK_VALUE else option.value,
                text='All' if option == ALL_SCOPES_CALLBACK_VALUE else format_selection_value(option),
            ),
            back_button=_fetch_back_button(step=MenuStep.SCOPE),
        ),
    )
    return True


async def _send_fetch_scopes(
    *,
    bot: Bot,
    chat_id: ChatId,
    services: Services,
    year: int,
    season: Season,
    universe: Universe,
    sub_season: SubSeason,
    scopes: Sequence[Scope],
) -> None:
    clip_group = ClipGroup(year=year, season=season, universe=universe)

    for index, scope in enumerate(scopes):
        if index > 0:
            await bot.send_message(chat_id=chat_id, text='.')

        async for batch in services.clip_store.fetch(
            clip_group=clip_group,
            clip_sub_group=ClipSubGroup(sub_season=sub_season, scope=scope),
            batch_size=10,
        ):

            await _send_stored_clip_batch(bot=bot, chat_id=chat_id, clips=batch)

    await bot.send_message(chat_id=chat_id, text='Done')


async def _send_stored_clip_batch(
    *,
    bot: Bot,
    chat_id: ChatId,
    clips: Sequence[Clip],
) -> None:
    if not clips:
        raise ValueError('`clips` must not be empty')

    if len(clips) == 1:
        clip = clips[0]
        await bot.send_video(
            chat_id=chat_id,
            video=BufferedInputFile(clip.bytes, filename=clip.filename),
        )
        return

    await bot.send_media_group(
        chat_id=chat_id,
        media=[
            InputMediaVideo(
                media=BufferedInputFile(clip.bytes, filename=clip.filename),
            )
            for clip in clips
        ],
    )


async def _fetch_sub_groups(
    *,
    services: Services,
    year: int,
    season: Season,
    universe: Universe,
) -> list[ClipSubGroup] | None:
    try:
        return await services.clip_store.list_sub_groups(ClipGroup(year=year, season=season, universe=universe))
    except ClipGroupNotFoundError:
        return None


def _fetch_year_options(groups: Sequence[ClipGroup]) -> list[int]:
    return sorted({group.year for group in groups})


def _fetch_season_options(
    groups: Sequence[ClipGroup],
    *,
    year: int,
) -> list[Season]:
    return [
        season
        for season in Season
        if any(group.year == year and group.season is season for group in groups)
    ]


def _fetch_universe_options(
    groups: Sequence[ClipGroup],
    *,
    year: int,
    season: Season,
) -> list[Universe]:
    return [
        universe
        for universe in Universe
        if any(group.year == year and group.season is season and group.universe is universe for group in groups)
    ]


def _fetch_sub_season_options(sub_groups: Sequence[ClipSubGroup]) -> list[SubSeason]:
    return [
        sub_season
        for sub_season in SubSeason
        if any(sub_group.sub_season is sub_season for sub_group in sub_groups)
    ]


def _fetch_scope_options(
    sub_groups: Sequence[ClipSubGroup],
    sub_season: SubSeason,
) -> list[Scope]:
    return [
        scope
        for scope in Scope
        if any(sub_group.sub_season is sub_season and sub_group.scope is scope for sub_group in sub_groups)
    ]


def _fetch_year_universe(settings: Settings) -> list[int]:
    current_year = date.today().year
    if current_year < settings.min_clip_year:
        return []
    return list(range(settings.min_clip_year, current_year + 1))


def _fetch_season_universe(year: int) -> list[Season]:
    return list(Season)


def _fetch_store_allowed_seasons(year: int) -> list[Season]:
    today = date.today()
    if year != today.year:
        return list(Season)
    max_season = Season.from_month(today.month)
    return [season for season in Season if season <= max_season]


def _fetch_entry_reply_markup():
    return stacked_keyboard(
        buttons=[
            InlineKeyboardButton(
                text='Fetch',
                callback_data=FetchEntryCallbackData(action=FetchEntryAction.OPEN).pack(),
            ),
            dummy_button(),
            InlineKeyboardButton(
                text='Cancel',
                callback_data=FetchEntryCallbackData(action=FetchEntryAction.CANCEL).pack(),
            ),
        ]
    )


async def _show_fetch_entry_menu(*, message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.edit_text(
        **width_reserved_text(
            text='Select action:',
            message_width=settings.message_width,
        ),
        reply_markup=_fetch_entry_reply_markup(),
    )


def _fetch_menu_button(*, step: MenuStep, value: str, text: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=text,
        callback_data=FetchCallbackData(
            action=MenuAction.SELECT,
            step=step,
            value=value,
        ).pack(),
    )


def _fetch_back_button(*, step: MenuStep) -> InlineKeyboardButton:
    return back_button(
        callback_data=FetchCallbackData(
            action=MenuAction.BACK,
            step=step,
            value=BACK_CALLBACK_VALUE,
        ).pack(),
    )


def _fetch_selection_labels(
    *,
    year: int | object = UNSET,
    season: Season | object = UNSET,
    universe: Universe | object = UNSET,
    sub_season: SubSeason | object = UNSET,
    scope: Scope | str | object = UNSET,
) -> list[str]:
    return [
        'Fetch',
        *selection_labels(
            year=year,
            season=season,
            universe=universe,
            sub_season=sub_season,
            scope=scope,
        ),
    ]
