from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from enum import StrEnum, auto

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.formatting import Text

from timeline_hub.handlers.clips.common import (
    ALL_SCOPES_CALLBACK_VALUE,
    FLOW_GET,
    FLOW_PULL,
    RETRIEVE_STATE_BY_STEP,
    MenuAction,
    MenuStep,
    encode_sub_season,
    parse_scope,
    parse_season,
    parse_sub_season,
    parse_universe,
    parse_year,
    set_flow_context,
)
from timeline_hub.handlers.clips.delivery import audio_normalization_from_settings, send_fetched_clip_batches
from timeline_hub.handlers.clips.flow import (
    FlowMenuDefinition,
    available_group_seasons,
    available_group_years,
    available_scopes,
    available_sub_seasons,
    flow_selection_labels,
    scope_option_callback_value,
    scope_option_text,
    selected_universe,
    selected_universe_year,
    selected_universe_year_season,
    selected_universe_year_season_sub_season,
    show_fixed_option_menu,
    show_or_stale,
    store_allowed_seasons,
    validate_menu_flow_state,
    year_option_universe,
)
from timeline_hub.handlers.menu import (
    callback_message,
    create_padding_line,
    handle_stale_selection,
    selected_text,
    selection_text,
    stacked_keyboard,
)
from timeline_hub.handlers.retrieve_common import StepOutcome
from timeline_hub.services.clip_store import (
    ClipGroup,
    ClipGroupNotFoundError,
    ClipSubGroup,
    Scope,
    Season,
    SubSeason,
    Universe,
)
from timeline_hub.services.container import Services
from timeline_hub.settings import Settings
from timeline_hub.types import ChatId

router = Router()
type BackStep = Callable[[], Awaitable[StepOutcome]]


class RetrieveEntryAction(StrEnum):
    GET = auto()
    PULL = auto()
    CANCEL = auto()


class RetrieveEntryCallbackData(CallbackData, prefix='clip_retrieve_entry'):
    action: RetrieveEntryAction


class RetrieveCallbackData(CallbackData, prefix='clip_retrieve'):
    action: MenuAction
    step: MenuStep
    value: str


def _pack_retrieve_menu_callback(action: MenuAction, step: MenuStep, value: str) -> str:
    return RetrieveCallbackData(action=action, step=step, value=value).pack()


async def _resolve_back_chain(*steps: BackStep, fallback: Callable[[], Awaitable[None]]) -> None:
    """Try back targets in order and fall back only if all request skip-back."""
    for step in steps:
        if await step() is StepOutcome.SHOWN:
            return
    await fallback()


_GET_FLOW = FlowMenuDefinition(
    mode=FLOW_GET,
    flow_label='Get',
    state_by_step=RETRIEVE_STATE_BY_STEP,
    pack_callback=_pack_retrieve_menu_callback,
)

_PULL_FLOW = FlowMenuDefinition(
    mode=FLOW_PULL,
    flow_label='Pull',
    state_by_step=RETRIEVE_STATE_BY_STEP,
    pack_callback=_pack_retrieve_menu_callback,
)


@router.message(F.text == 'Clips')
async def on_clips(message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.answer(
        **Text(
            create_padding_line(settings.message_width),
            '\n',
            'Select action:',
        ).as_kwargs(),
        reply_markup=_retrieve_entry_reply_markup(),
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
    bot: Bot | None = None,
) -> None:
    await callback.answer()
    message = callback_message(callback)
    if message is None:
        await state.clear()
        return

    if callback_data.action is RetrieveEntryAction.CANCEL:
        await state.clear()
        await message.edit_text(
            **selected_text(selected=['Cancel']),
            reply_markup=None,
        )
        return

    flow = _flow_for_entry_action(callback_data.action)
    if flow is None:
        await state.clear()
        return

    groups = await services.clip_store.list_groups()
    await state.update_data(groups=groups)
    await _show_retrieve_universe_menu(
        message=message,
        state=state,
        bot=bot,
        services=services,
        settings=settings,
        flow=flow,
    )


@router.callback_query(
    RetrieveCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_retrieve_menu(
    callback: CallbackQuery,
    callback_data: RetrieveCallbackData,
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

    data = await state.get_data()
    flow = _flow_for_mode(data.get('mode'))
    if flow is None:
        await handle_stale_selection(message=message, state=state)
        return

    if not await validate_menu_flow_state(
        message=message,
        state=state,
        flow=flow,
        step=callback_data.step,
    ):
        return

    if callback_data.action is MenuAction.BACK:
        await _on_retrieve_back(
            message=message,
            state=state,
            bot=bot,
            services=services,
            settings=settings,
            step=callback_data.step,
            flow=flow,
        )
        return

    await _on_retrieve_select(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        callback_data=callback_data,
        flow=flow,
    )


async def _on_retrieve_back(
    *,
    message: Message,
    state: FSMContext,
    bot: Bot,
    services: Services,
    settings: Settings,
    step: MenuStep,
    flow: FlowMenuDefinition,
) -> None:
    data = await state.get_data()
    groups = data.get('groups')
    if not isinstance(groups, list):
        await handle_stale_selection(message=message, state=state)
        return

    match step:
        case MenuStep.UNIVERSE:
            await _show_retrieve_entry_menu(message=message, state=state, settings=settings)

        case MenuStep.YEAR:
            await _resolve_back_chain(
                lambda: _show_retrieve_universe_menu(
                    message=message,
                    state=state,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                ),
                fallback=lambda: _show_retrieve_entry_menu(message=message, state=state, settings=settings),
            )

        case MenuStep.SEASON:
            selection = selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            await _resolve_back_chain(
                lambda: _show_retrieve_year_menu(
                    message=message,
                    state=state,
                    universe=universe,
                    bot=bot,
                    services=services,
                    year=year,
                    settings=settings,
                    flow=flow,
                ),
                lambda: _show_retrieve_universe_menu(
                    message=message,
                    state=state,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                ),
                fallback=lambda: _show_retrieve_entry_menu(message=message, state=state, settings=settings),
            )

        case MenuStep.SUB_SEASON:
            selection = selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            await _resolve_back_chain(
                lambda: _show_retrieve_season_menu(
                    message=message,
                    state=state,
                    universe=universe,
                    bot=bot,
                    services=services,
                    year=year,
                    settings=settings,
                    flow=flow,
                ),
                lambda: _show_retrieve_year_menu(
                    message=message,
                    state=state,
                    universe=universe,
                    bot=bot,
                    services=services,
                    year=year,
                    settings=settings,
                    flow=flow,
                ),
                lambda: _show_retrieve_universe_menu(
                    message=message,
                    state=state,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                ),
                fallback=lambda: _show_retrieve_entry_menu(message=message, state=state, settings=settings),
            )

        case MenuStep.SCOPE:
            selection = selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)

            sub_groups = await _retrieve_sub_groups(
                services=services,
                clip_group=clip_group,
            )
            if sub_groups is None:
                await handle_stale_selection(message=message, state=state)
                return

            if available_sub_seasons(sub_groups) == [SubSeason.NONE]:
                await _resolve_back_chain(
                    lambda: _show_retrieve_season_menu(
                        message=message,
                        state=state,
                        universe=universe,
                        bot=bot,
                        services=services,
                        year=year,
                        settings=settings,
                        flow=flow,
                    ),
                    lambda: _show_retrieve_year_menu(
                        message=message,
                        state=state,
                        universe=universe,
                        bot=bot,
                        services=services,
                        year=year,
                        settings=settings,
                        flow=flow,
                    ),
                    lambda: _show_retrieve_universe_menu(
                        message=message,
                        state=state,
                        bot=bot,
                        services=services,
                        settings=settings,
                        flow=flow,
                    ),
                    fallback=lambda: _show_retrieve_entry_menu(message=message, state=state, settings=settings),
                )
                return

            await _resolve_back_chain(
                lambda: _show_retrieve_sub_season_menu(
                    message=message,
                    state=state,
                    clip_group=clip_group,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                    sub_groups=sub_groups,
                ),
                lambda: _show_retrieve_season_menu(
                    message=message,
                    state=state,
                    universe=universe,
                    year=year,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                ),
                lambda: _show_retrieve_year_menu(
                    message=message,
                    state=state,
                    universe=universe,
                    bot=bot,
                    services=services,
                    year=year,
                    settings=settings,
                    flow=flow,
                ),
                lambda: _show_retrieve_universe_menu(
                    message=message,
                    state=state,
                    bot=bot,
                    services=services,
                    settings=settings,
                    flow=flow,
                ),
                fallback=lambda: _show_retrieve_entry_menu(message=message, state=state, settings=settings),
            )


async def _on_retrieve_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    callback_data: RetrieveCallbackData,
    flow: FlowMenuDefinition,
) -> None:
    data = await state.get_data()
    groups = data.get('groups')
    if not isinstance(groups, list):
        await handle_stale_selection(message=message, state=state)
        return

    match callback_data.step:
        case MenuStep.UNIVERSE:
            universe = parse_universe(callback_data.value)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            await show_or_stale(
                show_menu=_show_retrieve_year_menu,
                message=message,
                state=state,
                universe=universe,
                bot=bot,
                services=services,
                settings=settings,
                flow=flow,
            )

        case MenuStep.YEAR:
            universe = selected_universe(data)
            year = parse_year(callback_data.value)
            if universe is None or year is None:
                await handle_stale_selection(message=message, state=state)
                return
            await show_or_stale(
                show_menu=_show_retrieve_season_menu,
                message=message,
                state=state,
                universe=universe,
                bot=bot,
                services=services,
                year=year,
                settings=settings,
                flow=flow,
            )

        case MenuStep.SEASON:
            selection = selected_universe_year(data)
            season = parse_season(callback_data.value)
            if selection is None or season is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)
            await show_or_stale(
                show_menu=_show_retrieve_sub_season_menu,
                message=message,
                state=state,
                clip_group=clip_group,
                bot=bot,
                services=services,
                settings=settings,
                flow=flow,
            )

        case MenuStep.SUB_SEASON:
            selection = selected_universe_year_season(data)
            sub_season = parse_sub_season(callback_data.value)
            if selection is None or not isinstance(sub_season, SubSeason):
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)
            await show_or_stale(
                show_menu=_show_retrieve_scope_menu,
                message=message,
                state=state,
                clip_group=clip_group,
                sub_season=sub_season,
                bot=bot,
                services=services,
                settings=settings,
                flow=flow,
            )

        case MenuStep.SCOPE:
            selection = selected_universe_year_season_sub_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season, sub_season = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)

            sub_groups = await _retrieve_sub_groups(
                services=services,
                clip_group=clip_group,
            )
            if sub_groups is None:
                await handle_stale_selection(message=message, state=state)
                return

            scopes = available_scopes(sub_groups, sub_season)
            if not scopes:
                await handle_stale_selection(message=message, state=state)
                return

            if callback_data.value == ALL_SCOPES_CALLBACK_VALUE:
                selected_labels = flow_selection_labels(
                    flow,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    scope='All',
                )
            else:
                scope = parse_scope(callback_data.value)
                if scope is None or scope not in scopes:
                    await handle_stale_selection(message=message, state=state)
                    return
                scopes = [scope]
                selected_labels = flow_selection_labels(
                    flow,
                    universe=universe,
                    year=year,
                    season=season,
                    sub_season=sub_season,
                    scope=scope,
                )

            await message.edit_text(
                **selection_text(selected=selected_labels),
                reply_markup=None,
            )
            try:
                await _send_retrieve_scopes(
                    bot=bot,
                    chat_id=message.chat.id,
                    services=services,
                    clip_group=clip_group,
                    sub_season=sub_season,
                    scopes=scopes,
                    settings=settings,
                    normalize_audio=_normalizes_audio(flow),
                )
            except ClipGroupNotFoundError:
                await handle_stale_selection(message=message, state=state)
                return
            await state.clear()


async def _show_retrieve_year_menu(
    *,
    message: Message,
    state: FSMContext,
    universe: Universe,
    bot: Bot | None = None,
    services: Services | None = None,
    settings: Settings,
    flow: FlowMenuDefinition = _GET_FLOW,
    year: int | None = None,
) -> StepOutcome:
    data = await state.get_data()
    groups = data.get('groups')
    if not isinstance(groups, list):
        return StepOutcome.SKIP_BACK
    available_years = available_group_years(groups, universe=universe)
    if year is not None and year not in available_years:
        return StepOutcome.SKIP_BACK
    if len(available_years) == 1 and bot is not None and services is not None:
        selected_year = available_years[0]
        if selected_universe_year(data) == (universe, selected_year):
            return StepOutcome.SKIP_BACK
        await set_flow_context(
            state=state,
            mode=flow.mode,
            menu_message_id=message.message_id,
            fsm_state=flow.state_by_step[MenuStep.YEAR],
            universe=universe,
        )
        await _on_retrieve_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            callback_data=RetrieveCallbackData(
                action=MenuAction.SELECT,
                step=MenuStep.YEAR,
                value=str(selected_year),
            ),
            flow=flow,
        )
        return StepOutcome.SHOWN
    year_options = year_option_universe(current_year=date.today().year, min_year=settings.min_clip_year)

    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.YEAR,
        prompt='Select year:',
        universe=universe,
        option_universe=list(reversed(year_options)),
        available_options=available_years,
        option_value=str,
        option_text=str,
    )
    return StepOutcome.SHOWN


async def _show_retrieve_season_menu(
    *,
    message: Message,
    state: FSMContext,
    universe: Universe,
    year: int,
    bot: Bot | None = None,
    services: Services | None = None,
    settings: Settings,
    flow: FlowMenuDefinition = _GET_FLOW,
) -> StepOutcome:
    data = await state.get_data()
    groups = data.get('groups')
    if not isinstance(groups, list):
        return StepOutcome.SKIP_BACK
    available_seasons = available_group_seasons(groups, universe=universe, year=year)
    allowed_seasons = store_allowed_seasons(year=year, today=date.today())
    available_seasons = [season for season in available_seasons if season in allowed_seasons]
    if not available_seasons:
        return StepOutcome.SKIP_BACK
    if len(available_seasons) == 1 and bot is not None and services is not None:
        selected_season = available_seasons[0]
        if selected_universe_year_season(data) == (universe, year, selected_season):
            return StepOutcome.SKIP_BACK
        await set_flow_context(
            state=state,
            mode=flow.mode,
            menu_message_id=message.message_id,
            fsm_state=flow.state_by_step[MenuStep.SEASON],
            universe=universe,
            year=year,
        )
        await _on_retrieve_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            callback_data=RetrieveCallbackData(
                action=MenuAction.SELECT,
                step=MenuStep.SEASON,
                value=str(int(selected_season)),
            ),
            flow=flow,
        )
        return StepOutcome.SHOWN

    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.SEASON,
        prompt='Select season:',
        universe=universe,
        year=year,
        option_universe=list(Season),
        available_options=available_seasons,
        option_value=lambda season: str(int(season)),
        option_text=lambda season: str(int(season)),
    )
    return StepOutcome.SHOWN


async def _show_retrieve_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    bot: Bot | None = None,
    services: Services | None = None,
    settings: Settings,
    flow: FlowMenuDefinition = _GET_FLOW,
) -> StepOutcome:
    data = await state.get_data()
    groups = data.get('groups')
    if not isinstance(groups, list):
        return StepOutcome.SKIP_BACK
    available_universes = {group.universe for group in groups}
    available_options = [universe for universe in Universe if universe in available_universes]
    if len(available_options) == 1 and bot is not None and services is not None:
        selected_universe_value = available_options[0]
        if selected_universe(data) is selected_universe_value:
            return StepOutcome.SKIP_BACK
        await set_flow_context(
            state=state,
            mode=flow.mode,
            menu_message_id=message.message_id,
            fsm_state=flow.state_by_step[MenuStep.UNIVERSE],
        )
        await _on_retrieve_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            callback_data=RetrieveCallbackData(
                action=MenuAction.SELECT,
                step=MenuStep.UNIVERSE,
                value=selected_universe_value.value,
            ),
            flow=flow,
        )
        return StepOutcome.SHOWN
    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.UNIVERSE,
        prompt='Select universe:',
        option_universe=tuple(Universe),
        available_options=available_options,
        option_value=lambda universe: universe.value,
        option_text=lambda universe: universe.value.title(),
    )
    return StepOutcome.SHOWN


async def _show_retrieve_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    clip_group: ClipGroup,
    bot: Bot | None = None,
    services: Services,
    settings: Settings,
    flow: FlowMenuDefinition = _GET_FLOW,
    sub_groups: list[ClipSubGroup] | None = None,
) -> StepOutcome:
    data = await state.get_data()
    if sub_groups is None:
        sub_groups = await _retrieve_sub_groups(
            services=services,
            clip_group=clip_group,
        )
    if sub_groups is None:
        return StepOutcome.SKIP_BACK

    sub_seasons = available_sub_seasons(sub_groups)
    if sub_seasons == [SubSeason.NONE]:
        return await _show_retrieve_scope_menu(
            message=message,
            state=state,
            clip_group=clip_group,
            sub_season=SubSeason.NONE,
            bot=bot,
            services=services,
            settings=settings,
            sub_groups=sub_groups,
            flow=flow,
        )
    if len(sub_seasons) == 1 and bot is not None:
        selected_sub_season = sub_seasons[0]
        if selected_universe_year_season_sub_season(data) == (
            clip_group.universe,
            clip_group.year,
            clip_group.season,
            selected_sub_season,
        ):
            return StepOutcome.SKIP_BACK
        await set_flow_context(
            state=state,
            mode=flow.mode,
            menu_message_id=message.message_id,
            fsm_state=flow.state_by_step[MenuStep.SUB_SEASON],
            universe=clip_group.universe,
            year=clip_group.year,
            season=clip_group.season,
        )
        await _on_retrieve_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            callback_data=RetrieveCallbackData(
                action=MenuAction.SELECT,
                step=MenuStep.SUB_SEASON,
                value=encode_sub_season(selected_sub_season),
            ),
            flow=flow,
        )
        return StepOutcome.SHOWN
    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.SUB_SEASON,
        prompt='Select sub-season:',
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
        option_universe=tuple(SubSeason),
        available_options=sub_seasons,
        option_value=encode_sub_season,
        option_text=lambda sub_season: sub_season.value.title(),
    )
    return StepOutcome.SHOWN


async def _show_retrieve_scope_menu(
    *,
    message: Message,
    state: FSMContext,
    clip_group: ClipGroup,
    sub_season: SubSeason,
    bot: Bot | None = None,
    services: Services,
    settings: Settings,
    flow: FlowMenuDefinition = _GET_FLOW,
    sub_groups: list[ClipSubGroup] | None = None,
) -> StepOutcome:
    data = await state.get_data()
    if sub_groups is None:
        sub_groups = await _retrieve_sub_groups(
            services=services,
            clip_group=clip_group,
        )
    if sub_groups is None:
        return StepOutcome.SKIP_BACK

    scopes = available_scopes(sub_groups, sub_season)
    if not scopes:
        return StepOutcome.SKIP_BACK
    if len(scopes) == 1 and bot is not None:
        selected_scope = scopes[0]
        if (
            selected_universe_year_season_sub_season(data)
            == (clip_group.universe, clip_group.year, clip_group.season, sub_season)
            and await state.get_state() == flow.state_by_step[MenuStep.SCOPE].state
        ):
            return StepOutcome.SKIP_BACK
        await set_flow_context(
            state=state,
            mode=flow.mode,
            menu_message_id=message.message_id,
            fsm_state=flow.state_by_step[MenuStep.SCOPE],
            universe=clip_group.universe,
            year=clip_group.year,
            season=clip_group.season,
            sub_season=sub_season,
        )
        await _on_retrieve_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            bot=bot,
            callback_data=RetrieveCallbackData(
                action=MenuAction.SELECT,
                step=MenuStep.SCOPE,
                value=selected_scope.value,
            ),
            flow=flow,
        )
        return StepOutcome.SHOWN

    available_scope_options: list[Scope | str] = [ALL_SCOPES_CALLBACK_VALUE, *scopes]

    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.SCOPE,
        prompt='Select scope:',
        universe=clip_group.universe,
        year=clip_group.year,
        season=clip_group.season,
        sub_season=sub_season,
        option_universe=(ALL_SCOPES_CALLBACK_VALUE, *Scope),
        available_options=available_scope_options,
        option_value=scope_option_callback_value,
        option_text=scope_option_text,
    )
    return StepOutcome.SHOWN


async def _send_retrieve_scopes(
    *,
    bot: Bot,
    chat_id: ChatId,
    services: Services,
    clip_group: ClipGroup,
    sub_season: SubSeason,
    scopes: Sequence[Scope],
    settings: Settings,
    normalize_audio: bool,
) -> None:
    audio_normalization = audio_normalization_from_settings(settings=settings) if normalize_audio else None

    for index, scope in enumerate(scopes):
        if index > 0:
            await bot.send_message(chat_id=chat_id, text='.')

        await send_fetched_clip_batches(
            bot=bot,
            chat_id=chat_id,
            group=clip_group,
            sub_group=ClipSubGroup(sub_season=sub_season, scope=scope),
            clip_batches=services.clip_store.fetch(
                clip_group,
                ClipSubGroup(sub_season=sub_season, scope=scope),
                audio_normalization=audio_normalization,
            ),
        )


async def _retrieve_sub_groups(
    *,
    services: Services,
    clip_group: ClipGroup,
) -> list[ClipSubGroup] | None:
    try:
        return list((await services.clip_store.list_clips(clip_group)).keys())
    except ClipGroupNotFoundError:
        return None


def _retrieve_entry_reply_markup():
    return stacked_keyboard(
        buttons=[
            InlineKeyboardButton(
                text='Get',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.GET).pack(),
            ),
            InlineKeyboardButton(
                text='Pull',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.PULL).pack(),
            ),
            InlineKeyboardButton(
                text='Cancel',
                callback_data=RetrieveEntryCallbackData(action=RetrieveEntryAction.CANCEL).pack(),
            ),
        ]
    )


async def _show_retrieve_entry_menu(*, message: Message, state: FSMContext, settings: Settings) -> None:
    await state.clear()
    await message.edit_text(
        **Text(
            create_padding_line(settings.message_width),
            '\n',
            'Select action:',
        ).as_kwargs(),
        reply_markup=_retrieve_entry_reply_markup(),
    )


def _flow_for_entry_action(action: RetrieveEntryAction) -> FlowMenuDefinition | None:
    match action:
        case RetrieveEntryAction.GET:
            return _GET_FLOW
        case RetrieveEntryAction.PULL:
            return _PULL_FLOW
        case RetrieveEntryAction.CANCEL:
            return None


def _flow_for_mode(mode: object) -> FlowMenuDefinition | None:
    if mode == _GET_FLOW.mode:
        return _GET_FLOW
    if mode == _PULL_FLOW.mode:
        return _PULL_FLOW
    return None


def _normalizes_audio(flow: FlowMenuDefinition) -> bool:
    return flow is _GET_FLOW
