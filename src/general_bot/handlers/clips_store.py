import asyncio
from collections.abc import Sequence
from datetime import date
from enum import StrEnum, auto
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InputMediaVideo, Message
from aiogram.utils.formatting import Bold, Text

from general_bot.domain import normalize_video_volume
from general_bot.handlers.clips_common import (
    ALL_SCOPES_CALLBACK_VALUE,
    BACK_CALLBACK_VALUE,
    FLOW_STORE,
    STORE_STATE_BY_STEP,
    UNSET,
    MenuAction,
    MenuStep,
    StoreClipFlow,
    back_button,
    callback_message,
    create_padding_line,
    download_video_bytes,
    format_selection_value,
    format_store_summary,
    handle_stale_selection,
    parse_scope,
    parse_season,
    parse_sub_season,
    parse_universe,
    parse_year,
    selected_text,
    fixed_option_keyboard,
    selection_keyboard,
    selection_labels,
    selection_text,
    set_flow_context,
    split_sub_season_buttons,
    stacked_keyboard,
    validate_flow_state,
    width_reserved_text,
)
from general_bot.settings import Settings
from general_bot.services.clip_store import Clip, ClipGroup, ClipSubGroup, Scope, Season, StoreResult, SubSeason, Universe
from general_bot.services.container import Services
from general_bot.services.message_buffer import MessageGroup
from general_bot.types import ChatId

router = Router()


class ClipAction(StrEnum):
    NORMALIZE = auto()
    CANCEL = auto()
    STORE = auto()


class ClipActionCallbackData(CallbackData, prefix='clip_action'):
    action: ClipAction


class StoreCallbackData(CallbackData, prefix='clip_store'):
    action: MenuAction
    step: MenuStep
    value: str


@router.message(F.chat.type == ChatType.PRIVATE)
async def on_message_buffer_and_schedule_clip_action_selection(
    message: Message,
    services: Services,
    settings: Settings,
) -> None:
    chat_id = message.chat.id
    services.chat_message_buffer.append(message, chat_id=chat_id)

    async def send_clip_action_selection() -> None:
        kwargs = _clip_action_menu_kwargs(
            services=services,
            chat_id=chat_id,
            message_width=settings.message_width,
        )
        if kwargs is None:
            services.chat_message_buffer.flush(chat_id)
            await message.answer(text='No clips received')
            return
        await message.answer(**kwargs)

    services.task_scheduler.schedule(
        send_clip_action_selection,
        key=chat_id,
        delay=settings.forward_batch_timeout,
    )


@router.callback_query(
    ClipActionCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_clip_action(
    callback: CallbackQuery,
    callback_data: ClipActionCallbackData,
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

    match callback_data.action:
        case ClipAction.NORMALIZE:
            await state.clear()
            await message.edit_text(
                **selected_text(
                    selected=callback_data.action.title(),
                    leading_text=message.text or 'Clips',
                    message_width=settings.message_width,
                ),
                reply_markup=None,
            )
            await _normalize_buffered_clips(
                bot=bot,
                chat_id=message.chat.id,
                services=services,
                settings=settings,
            )

        case ClipAction.CANCEL:
            await state.clear()
            await message.edit_text(
                **selected_text(selected='Cancel'),
                reply_markup=None,
            )
            services.chat_message_buffer.flush(message.chat.id)
            await message.answer('Canceled')

        case ClipAction.STORE:
            await _show_store_year_menu(
                message=message,
                state=state,
                settings=settings,
            )


@router.callback_query(
    StoreCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_store_menu(
    callback: CallbackQuery,
    callback_data: StoreCallbackData,
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
        expected_mode=FLOW_STORE,
        expected_state=STORE_STATE_BY_STEP[callback_data.step],
    ):
        return

    if callback_data.action is MenuAction.BACK:
        await _on_store_back(
            message=message,
            state=state,
            services=services,
            settings=settings,
            step=callback_data.step,
        )
        return

    await _on_store_select(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        callback_data=callback_data,
    )


async def _normalize_buffered_clips(
    *,
    bot: Bot,
    chat_id: ChatId,
    services: Services,
    settings: Settings,
) -> None:
    message_groups = services.chat_message_buffer.flush_grouped(chat_id)
    cpu_semaphore = asyncio.Semaphore(1)

    async def normalize_message_clip_volume(message: Message) -> bytes | None:
        if message.video is None:
            return None

        video_bytes = await download_video_bytes(bot, file_id=message.video.file_id)

        # Limit concurrent CPU-bound video processing to avoid overloading the constrained runtime.
        async with cpu_semaphore:
            return await normalize_video_volume(
                video_bytes,
                loudness=settings.normalization_loudness,
                bitrate=settings.normalization_bitrate,
            )

    for message_group in message_groups:
        replacement_videos = await asyncio.wait_for(
            asyncio.gather(*(normalize_message_clip_volume(m) for m in message_group)),
            timeout=60,
        )
        await _resend_message_group(bot, chat_id, message_group, replacement_videos)


async def _on_store_back(
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
            await _show_clip_action_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
            )

        case MenuStep.SEASON:
            if not await _show_store_year_menu(message=message, state=state, settings=settings):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.UNIVERSE:
            year = data.get('year')
            if not isinstance(year, int):
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_store_season_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SUB_SEASON:
            year = data.get('year')
            season = data.get('season')
            if not isinstance(year, int) or not isinstance(season, Season):
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_store_universe_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
                season=season,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SCOPE:
            year = data.get('year')
            season = data.get('season')
            universe = data.get('universe')
            if not isinstance(year, int) or not isinstance(season, Season) or not isinstance(universe, Universe):
                await handle_stale_selection(message=message, state=state)
                return
            if not await _show_store_sub_season_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
                season=season,
                universe=universe,
            ):
                await handle_stale_selection(message=message, state=state)


async def _on_store_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    callback_data: StoreCallbackData,
) -> None:
    data = await state.get_data()

    match callback_data.step:
        case MenuStep.YEAR:
            year = parse_year(callback_data.value)
            if year is None or not await _show_store_season_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SEASON:
            year = data.get('year')
            season = parse_season(callback_data.value)
            if not isinstance(year, int) or season is None or not await _show_store_universe_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
                season=season,
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
                or not await _show_store_sub_season_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
                season=season,
                universe=universe,
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
                or not await _show_store_scope_menu(
                message=message,
                state=state,
                settings=settings,
                year=year,
                season=season,
                universe=universe,
                sub_season=sub_season,
            )
            ):
                await handle_stale_selection(message=message, state=state)

        case MenuStep.SCOPE:
            year = data.get('year')
            season = data.get('season')
            universe = data.get('universe')
            sub_season = data.get('sub_season', UNSET)
            scope = parse_scope(callback_data.value)
            if (
                not isinstance(year, int)
                or not isinstance(season, Season)
                or not isinstance(universe, Universe)
                or sub_season is UNSET
                or scope is None
            ):
                await handle_stale_selection(message=message, state=state)
                return

            await message.edit_text(
                **selection_text(
                    selected=_store_selection_labels(
                        year=year,
                        season=season,
                        universe=universe,
                        sub_season=sub_season,
                        scope=scope,
                    )
                ),
                reply_markup=None,
            )
            result = await _store_buffered_clips(
                bot=bot,
                chat_id=message.chat.id,
                services=services,
                year=year,
                season=season,
                universe=universe,
                sub_season=sub_season,
                scope=scope,
            )
            await state.clear()
            await message.answer(**_store_summary_kwargs(result))


async def _show_store_year_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
) -> bool:
    years = _store_year_options(current_year=date.today().year, min_year=settings.min_clip_year)
    if not years:
        return False

    await set_flow_context(
        state=state,
        mode=FLOW_STORE,
        menu_message_id=message.message_id,
        fsm_state=StoreClipFlow.year,
    )
    await message.edit_text(
        **selection_text(
            selected=_store_selection_labels(),
            prompt='Select year:',
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=list(reversed(years)),
            available_options=years,
            build_button=lambda year: _store_menu_button(
                step=MenuStep.YEAR,
                value=str(year),
                text=str(year),
            ),
            back_button=_store_back_button(step=MenuStep.YEAR),
        ),
    )
    return True


async def _show_store_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    year: int,
) -> bool:
    if year not in _store_year_options(current_year=date.today().year, min_year=settings.min_clip_year):
        return False
    seasons = _store_season_options(year=year, today=date.today())

    await set_flow_context(
        state=state,
        mode=FLOW_STORE,
        menu_message_id=message.message_id,
        fsm_state=StoreClipFlow.season,
        year=year,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select season:',
            selected=_store_selection_labels(year=year),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=list(Season),
            available_options=seasons,
            build_button=lambda season: _store_menu_button(
                step=MenuStep.SEASON,
                value=str(int(season)),
                text=str(int(season)),
            ),
            back_button=_store_back_button(step=MenuStep.SEASON),
        ),
    )
    return True


async def _show_store_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    year: int,
    season: Season,
) -> bool:
    await set_flow_context(
        state=state,
        mode=FLOW_STORE,
        menu_message_id=message.message_id,
        fsm_state=StoreClipFlow.universe,
        year=year,
        season=season,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select universe:',
            selected=_store_selection_labels(year=year, season=season),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(Universe),
            available_options=tuple(Universe),
            build_button=lambda universe: _store_menu_button(
                step=MenuStep.UNIVERSE,
                value=universe.value,
                text=format_selection_value(universe),
            ),
            back_button=_store_back_button(step=MenuStep.UNIVERSE),
        ),
    )
    return True


async def _show_store_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    year: int,
    season: Season,
    universe: Universe,
) -> bool:
    await set_flow_context(
        state=state,
        mode=FLOW_STORE,
        menu_message_id=message.message_id,
        fsm_state=StoreClipFlow.sub_season,
        year=year,
        season=season,
        universe=universe,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select sub-season:',
            selected=_store_selection_labels(year=year, season=season, universe=universe),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=tuple(SubSeason),
            available_options=tuple(SubSeason),
            build_button=lambda sub_season: _store_menu_button(
                step=MenuStep.SUB_SEASON,
                value=sub_season.value,
                text=format_selection_value(sub_season),
            ),
            back_button=_store_back_button(step=MenuStep.SUB_SEASON),
        ),
    )
    return True


async def _show_store_scope_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    year: int,
    season: Season,
    universe: Universe,
    sub_season: SubSeason,
) -> bool:
    await set_flow_context(
        state=state,
        mode=FLOW_STORE,
        menu_message_id=message.message_id,
        fsm_state=StoreClipFlow.scope,
        year=year,
        season=season,
        universe=universe,
        sub_season=sub_season,
    )
    await message.edit_text(
        **selection_text(
            prompt='Select scope:',
            selected=_store_selection_labels(
                year=year,
                season=season,
                universe=universe,
                sub_season=sub_season,
            ),
            message_width=settings.message_width,
        ),
        reply_markup=fixed_option_keyboard(
            option_universe=(ALL_SCOPES_CALLBACK_VALUE, *Scope),
            available_options=tuple(Scope),
            build_button=lambda option: _store_menu_button(
                step=MenuStep.SCOPE,
                value=ALL_SCOPES_CALLBACK_VALUE if option == ALL_SCOPES_CALLBACK_VALUE else option.value,
                text='All' if option == ALL_SCOPES_CALLBACK_VALUE else format_selection_value(option),
            ),
            back_button=_store_back_button(step=MenuStep.SCOPE),
        ),
    )
    return True


async def _store_buffered_clips(
    *,
    bot: Bot,
    chat_id: ChatId,
    services: Services,
    year: int,
    season: Season,
    universe: Universe,
    sub_season: SubSeason,
    scope: Scope,
) -> StoreResult:
    result = StoreResult(stored_count=0, duplicate_count=0)
    clip_group = ClipGroup(year=year, season=season, universe=universe)
    clip_sub_group = ClipSubGroup(sub_season=sub_season, scope=scope)
    message_groups = services.chat_message_buffer.flush_grouped(chat_id)

    for message_group in message_groups:
        clips = await _message_group_to_clips(bot=bot, message_group=message_group)
        if not clips:
            continue
        result += await services.clip_store.store(
            clips,
            clip_group=clip_group,
            clip_sub_group=clip_sub_group,
        )

    return result


async def _message_group_to_clips(
    *,
    bot: Bot,
    message_group: MessageGroup,
) -> list[Clip]:
    clips: list[Clip] = []

    for message in message_group:
        if message.video is None:
            continue
        clips.append(
            Clip(
                filename=_telegram_clip_filename(message),
                bytes=await download_video_bytes(bot, file_id=message.video.file_id),
            )
        )

    return clips


def _store_year_options(*, current_year: int, min_year: int) -> list[int]:
    if current_year < min_year:
        return []
    return list(range(min_year, current_year + 1))


def _store_season_options(*, year: int, today: date) -> list[Season]:
    if year != today.year:
        return list(Season)
    max_season = Season.from_month(today.month)
    return [season for season in Season if season <= max_season]


def _telegram_clip_filename(message: Message) -> str:
    if message.video is not None and message.video.file_name:
        return message.video.file_name
    return f'telegram-{message.chat.id}-{message.message_id}.mp4'


def _clip_action_menu_kwargs(
    *,
    services: Services,
    chat_id: ChatId,
    message_width: int,
) -> dict[str, Any] | None:
    clip_count = len([message for message in services.chat_message_buffer.peek(chat_id) if message.video is not None])
    if clip_count == 0:
        return None
    return {
        **Text(
            'Clips: ',
            Bold(str(clip_count)),
            '\n',
            create_padding_line(message_width),
            '\n',
            'Select action:',
        ).as_kwargs(),
        'reply_markup': stacked_keyboard(
            buttons=[
                _create_clip_action_button(ClipAction.STORE),
                _create_clip_action_button(ClipAction.NORMALIZE),
                _create_clip_action_button(ClipAction.CANCEL),
            ]
        ),
    }


async def _show_clip_action_menu(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
) -> None:
    await state.clear()
    kwargs = _clip_action_menu_kwargs(
        services=services,
        chat_id=message.chat.id,
        message_width=settings.message_width,
    )
    if kwargs is None:
        await message.edit_text('No clips received', reply_markup=None)
        return
    await message.edit_text(**kwargs)


async def _resend_message_group(
    bot: Bot,
    chat_id: ChatId,
    message_group: Sequence[Message],
    replacement_videos: Sequence[bytes | None] | None = None,
) -> None:
    if not message_group:
        raise ValueError('`message_group` must not be empty')
    if replacement_videos is None:
        replacement_videos = [None] * len(message_group)
    if len(replacement_videos) != len(message_group):
        raise ValueError('`replacement_videos` must have the same length as `message_group`')

    if len(message_group) == 1 and message_group[0].video is None:
        await bot.copy_message(
            chat_id=chat_id,
            from_chat_id=message_group[0].chat.id,
            message_id=message_group[0].message_id,
        )
        return

    if any(message.video is None for message in message_group):
        raise ValueError('Message group must contain only videos')

    media = []
    for message, replacement_video in zip(message_group, replacement_videos, strict=True):
        media.append(
            InputMediaVideo(
                media=(
                    message.video.file_id
                    if replacement_video is None
                    else BufferedInputFile(replacement_video, filename=message.video.file_name)
                ),
                caption=message.caption,
                caption_entities=message.caption_entities,
            )
        )

    await bot.send_media_group(chat_id=chat_id, media=media)


def _create_clip_action_button(action: ClipAction) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=action.title(),
        callback_data=ClipActionCallbackData(action=action).pack(),
    )


def _store_menu_button(*, step: MenuStep, value: str, text: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=text,
        callback_data=StoreCallbackData(
            action=MenuAction.SELECT,
            step=step,
            value=value,
        ).pack(),
    )


def _store_back_button(*, step: MenuStep) -> InlineKeyboardButton:
    return back_button(
        callback_data=StoreCallbackData(
            action=MenuAction.BACK,
            step=step,
            value=BACK_CALLBACK_VALUE,
        ).pack(),
    )


def _store_selection_labels(
    *,
    year: int | object = UNSET,
    season: Season | object = UNSET,
    universe: Universe | object = UNSET,
    sub_season: SubSeason | object = UNSET,
    scope: Scope | str | object = UNSET,
) -> list[str]:
    return [
        'Store',
        *selection_labels(
            year=year,
            season=season,
            universe=universe,
            sub_season=sub_season,
            scope=scope,
        ),
    ]


def _store_summary_kwargs(result: StoreResult) -> dict[str, Any]:
    summary = format_store_summary(result)
    if summary == 'Nothing changed':
        return {'text': summary}

    parts: list[object] = []
    for index, line in enumerate(summary.splitlines()):
        if index > 0:
            parts.append('\n')
        label, value = line.split(': ', maxsplit=1)
        parts.extend([f'{label}: ', Bold(value)])
    return Text(*parts).as_kwargs()
