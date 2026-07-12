import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum, auto
from typing import Any

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InputMediaVideo, Message
from aiogram.utils.formatting import Bold, Text
from loguru import logger

from timeline_hub.handlers.clips.common import (
    ALL_SCOPES_CALLBACK_VALUE,
    FLOW_RECONCILE,
    FLOW_STORE,
    RECONCILE_STATE_BY_STEP,
    STORE_STATE_BY_STEP,
    MenuAction,
    MenuStep,
    download_video_bytes,
    extract_clip_file_id,
    parse_scope,
    parse_season,
    parse_sub_season,
    parse_universe,
    parse_year,
    selection_labels,
    store_summary_kwargs,
)
from timeline_hub.handlers.clips.flow import (
    FlowMenuDefinition,
    flow_selection_labels,
    scope_option_callback_value,
    scope_option_text,
    selected_universe,
    selected_universe_year,
    selected_universe_year_season,
    selected_universe_year_season_sub_season,
    show_fixed_option_menu,
    store_allowed_seasons,
    validate_menu_flow_state,
    year_option_universe,
)
from timeline_hub.handlers.clips.reconcile_input import (
    clip_id_batch_count,
    parse_clip_identity_filename,
    prepare_reconcile_clip_id_batches,
)
from timeline_hub.handlers.clips.reorder_flow import (
    REORDER_FLOW_MODE,
    REORDER_RESET_CALLBACK_VALUE,
    ReorderCallbackData,
    ReorderClipFlow,
    parse_reorder_index,
    reorder_final_kwargs,
    reorder_selected_order_from_state,
    reorder_selection_keyboard,
    reorder_selection_kwargs,
    reorder_total_clips_from_state,
    reorder_validation_error,
    reordered_video_messages,
    show_reorder_selection_menu,
)
from timeline_hub.handlers.clips.route_planning import RouteBatch, parse_route_text, plan_route_batches
from timeline_hub.handlers.clips.store_execution import _uses_dense_layout, execute_store_or_produce
from timeline_hub.handlers.clips.transfer_planning import TransferBatch, plan_transfer_batches
from timeline_hub.handlers.menu import (
    callback_message,
    create_padding_line,
    dummy_button,
    handle_stale_selection,
    selected_text,
    selection_keyboard,
    selection_text,
    three_row_keyboard,
    validate_flow_state,
)
from timeline_hub.infra.ffmpeg import UnsupportedVideoCodecError
from timeline_hub.services.clip_store import (
    ClipGroup,
    ClipGroupNotFoundError,
    ClipId,
    ClipSubGroup,
    DuplicateClipIdsError,
    InvalidClipIdentityError,
    ReconcileResult,
    Scope,
    Season,
    StoreResult,
    SubSeason,
    TransferClipRef,
    TransferResult,
    Universe,
    UnknownClipsError,
)
from timeline_hub.services.container import Services
from timeline_hub.services.message_buffer import MessageGroup
from timeline_hub.settings import Settings
from timeline_hub.types import ChatId, Extension, FileBytes

router = Router()
_TELEGRAM_MEDIA_GROUP_LIMIT = 10
_ROUTE_STORE_CHUNK_SIZE = 8
_BUFFER_VERSION_KEY = 'buffer_version'
_PRODUCE_FLOW_MODE = 'produce'
_MIXED_GROUPS = 'mixed_groups'
type IntakeShowMenu = Callable[..., Awaitable[bool]]


class IntakeAction(StrEnum):
    CANCEL = auto()
    REORDER = auto()
    COMPACT = auto()
    ROUTE = auto()
    TRANSFER = auto()
    STORE = auto()
    PRODUCE = auto()
    REMOVE = auto()
    RECONCILE = auto()


class IntakeActionCallbackData(CallbackData, prefix='clip_action'):
    action: IntakeAction
    buffer_version: int


class IntakeCallbackData(CallbackData, prefix='clip_intake'):
    action: MenuAction
    step: MenuStep
    value: str


class RouteRequestKind(StrEnum):
    EXTERNAL = auto()
    INTERNAL = auto()
    MIXED = auto()


@dataclass(slots=True)
class _RouteResult:
    selection_groups: list[ClipGroup]
    store_result: StoreResult
    compact_targets: list[tuple[ClipGroup, SubSeason]]
    error_text: str | None = None


@dataclass(slots=True)
class _TransferExecutionResult:
    transfer_result: TransferResult


@dataclass(frozen=True, slots=True)
class _BufferedRouteClip:
    message: Message
    kind: RouteRequestKind


def _pack_intake_menu_callback(action: MenuAction, step: MenuStep, value: str) -> str:
    return IntakeCallbackData(action=action, step=step, value=value).pack()


_STORE_FLOW = FlowMenuDefinition(
    mode=FLOW_STORE,
    flow_label='Store',
    state_by_step=STORE_STATE_BY_STEP,
    pack_callback=_pack_intake_menu_callback,
)

_PRODUCE_FLOW = FlowMenuDefinition(
    mode=_PRODUCE_FLOW_MODE,
    flow_label='Produce',
    state_by_step=STORE_STATE_BY_STEP,
    pack_callback=_pack_intake_menu_callback,
)

_RECONCILE_FLOW = FlowMenuDefinition(
    mode=FLOW_RECONCILE,
    flow_label='Reconcile',
    state_by_step=RECONCILE_STATE_BY_STEP,
    pack_callback=_pack_intake_menu_callback,
)


@router.callback_query(
    IntakeActionCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_intake_action(
    callback: CallbackQuery,
    callback_data: IntakeActionCallbackData,
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

    callback_buffer_version = getattr(
        callback_data,
        'buffer_version',
        services.chat_message_buffer.version(message.chat.id),
    )
    if callback_buffer_version != services.chat_message_buffer.version(message.chat.id):
        await handle_stale_selection(message=message, state=state)
        return

    match callback_data.action:
        case IntakeAction.CANCEL:
            await state.clear()
            await message.edit_text(
                **selected_text(selected=['Cancel']),
                reply_markup=None,
            )
            services.chat_message_buffer.flush(message.chat.id)

        case IntakeAction.REORDER:
            video_messages = _buffered_video_messages(services.chat_message_buffer.peek_grouped(message.chat.id))
            if not video_messages:
                await state.clear()
                await message.edit_text('Selection is no longer available', reply_markup=None)
                return
            if (error_text := reorder_validation_error(len(video_messages))) is not None:
                await state.clear()
                # Invalid clip counts are treated as a hard rejection rather than
                # a valid interactive flow. We intentionally flush here to keep
                # the UI stateless and require the user to resend clips.
                services.chat_message_buffer.flush(message.chat.id)
                await message.edit_text(error_text, reply_markup=None)
                return

            await show_reorder_selection_menu(
                message=message,
                state=state,
                settings=settings,
                total_clips=len(video_messages),
                buffer_version=services.chat_message_buffer.version(message.chat.id),
            )

        case IntakeAction.COMPACT:
            chat_id = message.chat.id
            version_at_start = services.chat_message_buffer.version(chat_id)
            video_messages = _buffered_video_messages(services.chat_message_buffer.peek_grouped(chat_id))
            total_clips = len(video_messages)
            if total_clips == 0:
                await state.clear()
                await message.edit_text('Selection is no longer available', reply_markup=None)
                return
            if total_clips == 1:
                await state.clear()
                services.chat_message_buffer.flush(chat_id)
                await message.edit_text('Unexpected number of clips', reply_markup=None)
                return

            # `0` clips means the action menu went stale, so we keep the buffer.
            # `1` clip is an invalid compact input, so we flush intentionally.
            if services.chat_message_buffer.version(chat_id) != version_at_start:
                await state.clear()
                await message.edit_text('Selection is no longer available', reply_markup=None)
                return

            await state.clear()
            await message.edit_text(
                **selected_text(selected='Compact'),
                reply_markup=None,
            )
            compact_message_groups = services.chat_message_buffer.peek_grouped(chat_id)
            services.chat_message_buffer.flush(chat_id)
            compact_messages = _buffered_video_messages(compact_message_groups)
            await _send_reordered_video_messages(
                bot=bot,
                chat_id=chat_id,
                messages=compact_messages,
            )

        case IntakeAction.RECONCILE:
            if _has_buffered_videos(
                services=services,
                chat_id=message.chat.id,
            ):
                try:
                    clip_group, clip_id_batches = _pending_reconcile_clip_id_batches(
                        services=services,
                        chat_id=message.chat.id,
                    )
                except DuplicateClipIdsError:
                    await _invalidate_intake_buffer(
                        message=message,
                        state=state,
                        services=services,
                        text="Can't reconcile duplicates",
                    )
                    return
                except ValueError as error:
                    text = (
                        "Can't reconcile mixed groups" if str(error) == _MIXED_GROUPS else "Can't reconcile not stored"
                    )
                    await _invalidate_intake_buffer(
                        message=message,
                        state=state,
                        services=services,
                        text=text,
                    )
                    return

                buffer_version = services.chat_message_buffer.version(message.chat.id)
                await _show_reconcile_sub_season_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    clip_group=clip_group,
                    clip_id_batches=clip_id_batches,
                    buffer_version=buffer_version,
                )
                return

            stored_data = await state.get_data()
            stored_clip_group = _reconcile_clip_group_from_state(stored_data)
            stored_clip_id_batches = _reconcile_clip_id_batches_from_state(stored_data)
            if stored_clip_group is not None and stored_clip_id_batches is not None:
                await _show_reconcile_sub_season_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    clip_group=stored_clip_group,
                    clip_id_batches=stored_clip_id_batches,
                )
                return

            await handle_stale_selection(message=message, state=state)

        case IntakeAction.STORE:
            await _show_intake_menu_or_stale(
                show_menu=_show_store_universe_menu,
                message=message,
                state=state,
                buffer_version=services.chat_message_buffer.version(message.chat.id),
                settings=settings,
                flow=_STORE_FLOW,
            )

        case IntakeAction.PRODUCE:
            await _show_intake_menu_or_stale(
                show_menu=_show_store_universe_menu,
                message=message,
                state=state,
                buffer_version=services.chat_message_buffer.version(message.chat.id),
                settings=settings,
                flow=_PRODUCE_FLOW,
            )

        case IntakeAction.REMOVE:
            chat_id = message.chat.id
            buffered_video_messages = _buffered_video_messages(services.chat_message_buffer.peek_grouped(chat_id))
            if not buffered_video_messages:
                await handle_stale_selection(message=message, state=state)
                return

            clip_group: ClipGroup | None = None
            clip_ids: list[ClipId] = []
            clip_id_set: set[ClipId] = set()
            try:
                for buffered_message in buffered_video_messages:
                    if buffered_message.video is None or not buffered_message.video.file_name:
                        raise InvalidClipIdentityError('clip filename is required')
                    parsed_group, clip_id = parse_clip_identity_filename(buffered_message.video.file_name)
                    if clip_group is None:
                        clip_group = parsed_group
                    elif parsed_group != clip_group:
                        raise ValueError(_MIXED_GROUPS)
                    if clip_id in clip_id_set:
                        raise DuplicateClipIdsError(clip_ids=[*clip_ids, clip_id])
                    clip_ids.append(clip_id)
                    clip_id_set.add(clip_id)
            except DuplicateClipIdsError, InvalidClipIdentityError, ValueError:
                await _invalidate_intake_buffer(
                    message=message,
                    state=state,
                    services=services,
                    text='Invalid input',
                )
                return

            if clip_group is None:
                await handle_stale_selection(message=message, state=state)
                return

            services.chat_message_buffer.flush(chat_id)
            await state.clear()
            await message.edit_text(
                **selected_text(selected='Remove'),
                reply_markup=None,
            )

            try:
                affected_sub_groups = await services.clip_store.remove(
                    clip_group,
                    clip_ids=clip_ids,
                )
            except ClipGroupNotFoundError, UnknownClipsError:
                await _invalidate_intake_buffer(
                    message=message,
                    state=state,
                    services=services,
                    text='Invalid input',
                )
                return

            for sub_group in affected_sub_groups:
                if not _uses_dense_layout(sub_group.scope):
                    continue
                try:
                    await services.clip_store.compact(
                        clip_group,
                        sub_group,
                        batch_size=_TELEGRAM_MEDIA_GROUP_LIMIT,
                        require_exists=False,
                    )
                except Exception:
                    logger.exception(
                        'post-remove clip compaction failed for {} {}',
                        clip_group,
                        sub_group,
                    )
                    raise

            await message.answer('Done')

        case IntakeAction.ROUTE:
            await state.clear()
            # Route is a single-shot action: flush at entry, validate after flush,
            # never restore the buffer on failure. This is intentional to keep the
            # UI stateless and simple; users must resend clips if validation fails.
            route_message_groups = services.chat_message_buffer.peek_grouped(message.chat.id)
            services.chat_message_buffer.flush(message.chat.id)
            route_kind, route_clip_messages = _classify_route_request(route_message_groups)
            if route_kind is RouteRequestKind.MIXED:
                await message.edit_text('Mixed clip types', reply_markup=None)
                return
            if route_kind is RouteRequestKind.INTERNAL:
                await _execute_internal_route_action(
                    message=message,
                    services=services,
                    settings=settings,
                    state=state,
                    transfer_message_groups=route_message_groups,
                )
                return

            route_batches, error_text = plan_route_batches(route_message_groups, settings=settings)
            if error_text is not None:
                await message.edit_text(error_text, reply_markup=None)
                return
            if not route_batches or not route_clip_messages:
                await message.edit_text('No clips received', reply_markup=None)
                return

            await message.edit_text('Routing...', reply_markup=None)

            async def update_route_progress(selection_batches: Sequence[RouteBatch]) -> None:
                await message.edit_text(
                    **_route_progress_kwargs(selection_batches),
                    reply_markup=None,
                )

            try:
                route_result = await _store_route_batches(
                    bot=bot,
                    services=services,
                    route_batches=route_batches,
                    on_batch_stored=update_route_progress,
                )
            except UnsupportedVideoCodecError:
                await message.edit_text('Invalid codec', reply_markup=None)
                return
            for clip_group, sub_season in route_result.compact_targets:
                for clip_sub_group in (
                    ClipSubGroup(sub_season=sub_season, scope=Scope.SOURCE),
                    ClipSubGroup(sub_season=sub_season, scope=Scope.EXTRA),
                ):
                    try:
                        await services.clip_store.compact(
                            clip_group,
                            clip_sub_group,
                            batch_size=_TELEGRAM_MEDIA_GROUP_LIMIT,
                            require_exists=clip_sub_group.scope is Scope.SOURCE,
                        )
                    except Exception:
                        logger.exception(
                            'post-store clip compaction failed for {} {}',
                            clip_group,
                            clip_sub_group,
                        )
                        raise
            await message.answer(**store_summary_kwargs(route_result.store_result))

        case IntakeAction.TRANSFER:
            await state.clear()
            transfer_message_groups = services.chat_message_buffer.peek_grouped(message.chat.id)
            services.chat_message_buffer.flush(message.chat.id)
            await _execute_internal_route_action(
                message=message,
                services=services,
                settings=settings,
                state=state,
                transfer_message_groups=transfer_message_groups,
            )


@router.callback_query(
    ReorderCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_reorder_menu(
    callback: CallbackQuery,
    callback_data: ReorderCallbackData,
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
        expected_mode=REORDER_FLOW_MODE,
        expected_state=ReorderClipFlow.selecting,
    ):
        return

    if callback_data.action is MenuAction.BACK:
        if callback_data.value == 'back':
            await _show_intake_action_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
            )
            return

        data = await state.get_data()
        selected_order = reorder_selected_order_from_state(data)
        total_clips = reorder_total_clips_from_state(data)
        if (
            callback_data.value != REORDER_RESET_CALLBACK_VALUE
            or selected_order is None
            or total_clips is None
            or not selected_order
        ):
            await handle_stale_selection(message=message, state=state)
            return

        if not _is_intake_buffer_state_valid(
            data=data,
            services=services,
            chat_id=message.chat.id,
        ):
            await handle_stale_selection(message=message, state=state)
            return

        await state.update_data(selected_order=[])
        await message.edit_text(
            **reorder_selection_kwargs(
                selected_order=[],
                message_width=settings.message_width,
            ),
            reply_markup=reorder_selection_keyboard(
                total_clips=total_clips,
                selected_order=[],
            ),
        )
        return

    index = parse_reorder_index(callback_data.value)
    data = await state.get_data()
    selected_order = reorder_selected_order_from_state(data)
    total_clips = reorder_total_clips_from_state(data)
    if index is None or selected_order is None or total_clips is None:
        await handle_stale_selection(message=message, state=state)
        return

    if not _is_intake_buffer_state_valid(
        data=data,
        services=services,
        chat_id=message.chat.id,
    ):
        await handle_stale_selection(message=message, state=state)
        return

    if index < 1 or index > total_clips:
        await handle_stale_selection(message=message, state=state)
        return

    if index in selected_order:
        return

    updated_order = [*selected_order, index]
    if len(updated_order) == total_clips:
        await message.edit_text(
            **reorder_final_kwargs(updated_order),
            reply_markup=None,
        )
        reordered_message_groups = services.chat_message_buffer.peek_grouped(message.chat.id)
        services.chat_message_buffer.flush(message.chat.id)
        reordered_messages = reordered_video_messages(
            _buffered_video_messages(reordered_message_groups),
            selected_order=updated_order,
            total_clips=total_clips,
        )
        await _send_reordered_video_messages(
            bot=bot,
            chat_id=message.chat.id,
            messages=reordered_messages,
        )
        await state.clear()
        return

    await state.update_data(selected_order=updated_order)
    await message.edit_text(
        **reorder_selection_kwargs(
            selected_order=updated_order,
            message_width=settings.message_width,
        ),
        reply_markup=reorder_selection_keyboard(
            total_clips=total_clips,
            selected_order=updated_order,
        ),
    )


@router.callback_query(
    IntakeCallbackData.filter(),
    F.message.chat.type == ChatType.PRIVATE,
)
async def on_intake_menu(
    callback: CallbackQuery,
    callback_data: IntakeCallbackData,
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
    flow = _selection_flow_for_mode(data.get('mode'))
    if flow is None:
        await handle_stale_selection(message=message, state=state)
        return

    if callback_data.step not in flow.state_by_step:
        await handle_stale_selection(message=message, state=state)
        return

    if not await validate_menu_flow_state(
        message=message,
        state=state,
        flow=flow,
        step=callback_data.step,
    ):
        return

    if not _is_intake_buffer_state_valid(
        data=data,
        services=services,
        chat_id=message.chat.id,
    ):
        await handle_stale_selection(message=message, state=state)
        return

    if callback_data.action is MenuAction.BACK:
        await _on_store_back(
            message=message,
            state=state,
            services=services,
            settings=settings,
            step=callback_data.step,
            flow=flow,
        )
        return

    await _on_store_select(
        message=message,
        state=state,
        services=services,
        settings=settings,
        bot=bot,
        callback_data=callback_data,
        flow=flow,
    )


async def _on_store_back(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    step: MenuStep,
    flow: FlowMenuDefinition,
) -> None:
    data = await state.get_data()

    if flow is _RECONCILE_FLOW:
        match step:
            case MenuStep.SUB_SEASON:
                clip_id_batches = _reconcile_clip_id_batches_from_state(data)
                if clip_id_batches is None:
                    await handle_stale_selection(message=message, state=state)
                    return
                await _show_intake_action_menu(
                    message=message,
                    state=state,
                    services=services,
                    settings=settings,
                    clip_count_override=clip_id_batch_count(clip_id_batches),
                    preserve_state=True,
                )

            case MenuStep.SCOPE:
                clip_group = _reconcile_clip_group_from_state(data)
                clip_id_batches = _reconcile_clip_id_batches_from_state(data)
                if clip_group is None or clip_id_batches is None:
                    await handle_stale_selection(message=message, state=state)
                    return
                await _show_reconcile_sub_season_menu(
                    message=message,
                    state=state,
                    settings=settings,
                    clip_group=clip_group,
                    clip_id_batches=clip_id_batches,
                )
        return

    match step:
        case MenuStep.UNIVERSE:
            await _show_intake_action_menu(
                message=message,
                state=state,
                services=services,
                settings=settings,
            )

        case MenuStep.YEAR:
            await _show_intake_menu_or_stale(
                show_menu=_show_store_universe_menu,
                message=message,
                state=state,
                settings=settings,
                flow=flow,
            )

        case MenuStep.SEASON:
            selection = selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            await _show_intake_menu_or_stale(
                show_menu=_show_store_year_menu,
                message=message,
                state=state,
                universe=universe,
                settings=settings,
                year=year,
                flow=flow,
            )

        case MenuStep.SUB_SEASON:
            selection = selected_universe_year(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            await _show_intake_menu_or_stale(
                show_menu=_show_store_season_menu,
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                year=year,
                flow=flow,
            )

        case MenuStep.SCOPE:
            selection = selected_universe_year_season(data)
            if selection is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)
            await _show_intake_menu_or_stale(
                show_menu=_show_store_sub_season_menu,
                message=message,
                state=state,
                settings=settings,
                clip_group=clip_group,
                flow=flow,
            )


async def _on_store_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    bot: Bot,
    callback_data: IntakeCallbackData,
    flow: FlowMenuDefinition,
) -> None:
    data = await state.get_data()

    if flow is _RECONCILE_FLOW:
        await _on_reconcile_select(
            message=message,
            state=state,
            services=services,
            settings=settings,
            callback_data=callback_data,
        )
        return

    match callback_data.step:
        case MenuStep.UNIVERSE:
            universe = parse_universe(callback_data.value)
            if universe is None:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_intake_menu_or_stale(
                show_menu=_show_store_year_menu,
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                flow=flow,
            )

        case MenuStep.YEAR:
            universe = selected_universe(data)
            year = parse_year(callback_data.value)
            if universe is None or year is None:
                await handle_stale_selection(message=message, state=state)
                return
            await _show_intake_menu_or_stale(
                show_menu=_show_store_season_menu,
                message=message,
                state=state,
                settings=settings,
                universe=universe,
                year=year,
                flow=flow,
            )

        case MenuStep.SEASON:
            selection = selected_universe_year(data)
            season = parse_season(callback_data.value)
            if selection is None or season is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year = selection
            if season not in _store_season_options(year=year, today=date.today()):
                await handle_stale_selection(message=message, state=state)
                return
            clip_group = ClipGroup(universe=universe, year=year, season=season)
            await _show_intake_menu_or_stale(
                show_menu=_show_store_sub_season_menu,
                message=message,
                state=state,
                settings=settings,
                clip_group=clip_group,
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
            await _show_intake_menu_or_stale(
                show_menu=_show_store_scope_menu,
                message=message,
                state=state,
                settings=settings,
                clip_group=clip_group,
                sub_season=sub_season,
                flow=flow,
            )

        case MenuStep.SCOPE:
            selection = selected_universe_year_season_sub_season(data)
            scope = parse_scope(callback_data.value)
            if selection is None or scope is None:
                await handle_stale_selection(message=message, state=state)
                return
            universe, year, season, sub_season = selection
            clip_group = ClipGroup(universe=universe, year=year, season=season)
            clip_sub_group = ClipSubGroup(sub_season=sub_season, scope=scope)

            if flow is _STORE_FLOW or flow is _PRODUCE_FLOW:
                await execute_store_or_produce(
                    bot=bot,
                    message=message,
                    state=state,
                    services=services,
                    settings=settings,
                    clip_group=clip_group,
                    clip_sub_group=clip_sub_group,
                    selection_kwargs=selection_text(
                        selected=flow_selection_labels(
                            flow,
                            universe=universe,
                            year=year,
                            season=season,
                            sub_season=sub_season,
                            scope=scope,
                        )
                    ),
                    produce=flow is _PRODUCE_FLOW,
                )
                return


async def _on_reconcile_select(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    callback_data: IntakeCallbackData,
) -> None:
    data = await state.get_data()
    clip_group = _reconcile_clip_group_from_state(data)
    clip_id_batches = _reconcile_clip_id_batches_from_state(data)
    if clip_group is None or clip_id_batches is None:
        await handle_stale_selection(message=message, state=state)
        return

    match callback_data.step:
        case MenuStep.SUB_SEASON:
            sub_season = parse_sub_season(callback_data.value)
            if not isinstance(sub_season, SubSeason):
                await handle_stale_selection(message=message, state=state)
                return
            await _show_reconcile_scope_menu(
                message=message,
                state=state,
                settings=settings,
                clip_group=clip_group,
                sub_season=sub_season,
                clip_id_batches=clip_id_batches,
            )

        case MenuStep.SCOPE:
            sub_season = data.get('sub_season')
            scope = parse_scope(callback_data.value)
            if not isinstance(sub_season, SubSeason) or scope is None:
                await handle_stale_selection(message=message, state=state)
                return
            clip_sub_group = ClipSubGroup(sub_season=sub_season, scope=scope)

            await message.edit_text(
                **selection_text(
                    selected=flow_selection_labels(
                        _RECONCILE_FLOW,
                        universe=clip_group.universe,
                        year=clip_group.year,
                        season=clip_group.season,
                        sub_season=sub_season,
                        scope=scope,
                    )
                ),
                reply_markup=None,
            )
            services.chat_message_buffer.flush(message.chat.id)
            await state.clear()

            try:
                result = await services.clip_store.reconcile(
                    clip_group,
                    clip_sub_group,
                    clip_id_batches=clip_id_batches,
                )
            except DuplicateClipIdsError:
                await message.answer(text="Can't reconcile duplicates")
                return
            except UnknownClipsError:
                await message.answer(text="Can't reconcile not stored")
                return
            await message.answer(**_reconcile_summary_kwargs(result))

        case _:
            await handle_stale_selection(message=message, state=state)


async def _show_store_year_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    universe: Universe,
    year: int | None = None,
    flow: FlowMenuDefinition = _STORE_FLOW,
) -> bool:
    years = _store_year_options(current_year=date.today().year, min_year=settings.min_year)
    if not years:
        return False
    if year is not None and year not in years:
        return False

    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.YEAR,
        prompt='Select year:',
        universe=universe,
        option_universe=list(reversed(years)),
        available_options=years,
        option_value=str,
        option_text=str,
    )
    return True


async def _show_store_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    universe: Universe,
    year: int,
    flow: FlowMenuDefinition = _STORE_FLOW,
) -> bool:
    if year not in _store_year_options(current_year=date.today().year, min_year=settings.min_year):
        return False
    seasons = _store_season_options(year=year, today=date.today())

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
        available_options=seasons,
        option_value=lambda season: str(int(season)),
        option_text=lambda season: str(int(season)),
    )
    return True


async def _show_store_universe_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    flow: FlowMenuDefinition = _STORE_FLOW,
) -> bool:
    await show_fixed_option_menu(
        flow=flow,
        message=message,
        state=state,
        message_width=settings.message_width,
        step=MenuStep.UNIVERSE,
        prompt='Select universe:',
        option_universe=tuple(Universe),
        available_options=tuple(Universe),
        option_value=lambda universe: universe.value,
        option_text=lambda universe: universe.value.title(),
    )
    return True


async def _show_store_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    clip_group: ClipGroup,
    flow: FlowMenuDefinition = _STORE_FLOW,
) -> bool:
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
        available_options=tuple(SubSeason),
        option_value=lambda sub_season: sub_season.value,
        option_text=lambda sub_season: sub_season.value.title(),
    )
    return True


async def _show_store_scope_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    clip_group: ClipGroup,
    sub_season: SubSeason,
    flow: FlowMenuDefinition = _STORE_FLOW,
) -> bool:
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
        available_options=tuple(Scope),
        option_value=scope_option_callback_value,
        option_text=scope_option_text,
    )
    return True


async def _show_reconcile_sub_season_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    clip_group: ClipGroup,
    clip_id_batches: list[list[ClipId]],
    buffer_version: int | None = None,
) -> None:
    if not await _show_intake_menu_or_stale(
        show_menu=_show_store_sub_season_menu,
        message=message,
        state=state,
        buffer_version=buffer_version,
        settings=settings,
        clip_group=clip_group,
        flow=_RECONCILE_FLOW,
    ):
        return
    await state.update_data(
        clip_group=clip_group,
        clip_id_batches=clip_id_batches,
    )


async def _show_reconcile_scope_menu(
    *,
    message: Message,
    state: FSMContext,
    settings: Settings,
    clip_group: ClipGroup,
    sub_season: SubSeason,
    clip_id_batches: list[list[ClipId]],
) -> None:
    if not await _show_intake_menu_or_stale(
        show_menu=_show_store_scope_menu,
        message=message,
        state=state,
        settings=settings,
        clip_group=clip_group,
        sub_season=sub_season,
        flow=_RECONCILE_FLOW,
    ):
        return
    await state.update_data(
        clip_group=clip_group,
        clip_id_batches=clip_id_batches,
    )


async def _store_route_batches(
    *,
    bot: Bot,
    services: Services,
    route_batches: Sequence[RouteBatch],
    on_batch_stored: Callable[[Sequence[RouteBatch]], Awaitable[None]] | None = None,
) -> _RouteResult:
    result = StoreResult(stored_count=0, duplicate_count=0)
    compact_targets: list[tuple[ClipGroup, SubSeason]] = []
    compact_target_set: set[tuple[ClipGroup, SubSeason]] = set()
    completed_route_batches: list[RouteBatch] = []
    selection_groups: list[ClipGroup] = []

    for route_batch in route_batches:
        stored_any = False
        for start in range(0, len(route_batch.messages), _ROUTE_STORE_CHUNK_SIZE):
            batch_messages = route_batch.messages[start : start + _ROUTE_STORE_CHUNK_SIZE]
            try:
                batch_result = await services.clip_store.store(
                    route_batch.clip_group,
                    ClipSubGroup(sub_season=route_batch.sub_season, scope=Scope.SOURCE),
                    clips=await _clip_messages_to_clip_files(
                        bot=bot,
                        messages=batch_messages,
                    ),
                )
            except UnsupportedVideoCodecError as error:
                _log_route_unsupported_codec_warning(
                    error=error,
                    chat_id=batch_messages[0].chat.id,
                    clip_group=route_batch.clip_group,
                    sub_season=route_batch.sub_season,
                    messages=batch_messages,
                )
                raise
            result += batch_result
            if batch_result.stored_count > 0:
                stored_any = True

        completed_route_batches.append(route_batch)
        selection_groups.append(route_batch.clip_group)
        if on_batch_stored is not None:
            await on_batch_stored(completed_route_batches)
        compact_target = (route_batch.clip_group, route_batch.sub_season)
        if stored_any and compact_target not in compact_target_set:
            compact_targets.append(compact_target)
            compact_target_set.add(compact_target)

    return _RouteResult(
        selection_groups=selection_groups,
        store_result=result,
        compact_targets=compact_targets,
    )


async def _transfer_clip_batches(
    *,
    services: Services,
    transfer_batches: Sequence[TransferBatch],
    on_batch_transferred: Callable[[Sequence[TransferBatch]], Awaitable[None]] | None = None,
) -> _TransferExecutionResult:
    result = TransferResult(
        transferred_count=0,
        already_in_destination_group_count=0,
        duplicate_blocked_count=0,
    )
    completed_transfer_batches: list[TransferBatch] = []
    affected_sub_groups: list[tuple[ClipGroup, ClipSubGroup]] = []
    affected_sub_group_set: set[tuple[ClipGroup, ClipSubGroup]] = set()

    for transfer_batch in transfer_batches:
        batch_result = await services.clip_store.transfer(
            destination_group=transfer_batch.destination_group,
            clips=[
                TransferClipRef(
                    source_group=clip.source_group,
                    clip_id=clip.clip_id,
                )
                for clip in transfer_batch.clips
            ],
        )
        result = TransferResult(
            transferred_count=result.transferred_count + batch_result.transferred_count,
            already_in_destination_group_count=(
                result.already_in_destination_group_count + batch_result.already_in_destination_group_count
            ),
            duplicate_blocked_count=result.duplicate_blocked_count + batch_result.duplicate_blocked_count,
        )
        for affected_sub_group in batch_result.affected_sub_groups:
            if affected_sub_group in affected_sub_group_set:
                continue
            affected_sub_group_set.add(affected_sub_group)
            affected_sub_groups.append(affected_sub_group)
        completed_transfer_batches.append(transfer_batch)
        if on_batch_transferred is not None:
            await on_batch_transferred(completed_transfer_batches)

    return _TransferExecutionResult(
        transfer_result=TransferResult(
            transferred_count=result.transferred_count,
            already_in_destination_group_count=result.already_in_destination_group_count,
            duplicate_blocked_count=result.duplicate_blocked_count,
            affected_sub_groups=tuple(affected_sub_groups),
        )
    )


def _classify_route_request(
    message_groups: Sequence[MessageGroup],
) -> tuple[RouteRequestKind, list[_BufferedRouteClip]]:
    buffered_clips: list[_BufferedRouteClip] = []
    seen_kinds: set[RouteRequestKind] = set()

    for message_group in message_groups:
        for message in message_group:
            if extract_clip_file_id(message) is None:
                continue
            route_kind = _route_clip_kind(message)
            seen_kinds.add(route_kind)
            buffered_clips.append(_BufferedRouteClip(message=message, kind=route_kind))

    if RouteRequestKind.EXTERNAL in seen_kinds and RouteRequestKind.INTERNAL in seen_kinds:
        return RouteRequestKind.MIXED, buffered_clips
    if RouteRequestKind.INTERNAL in seen_kinds:
        return RouteRequestKind.INTERNAL, buffered_clips
    return RouteRequestKind.EXTERNAL, buffered_clips


def _route_clip_kind(message: Message) -> RouteRequestKind:
    file_name = _route_message_filename(message)
    if not file_name:
        return RouteRequestKind.EXTERNAL
    try:
        parse_clip_identity_filename(file_name)
    except ValueError:
        return RouteRequestKind.EXTERNAL
    return RouteRequestKind.INTERNAL


async def _execute_internal_route_action(
    *,
    message: Message,
    services: Services,
    settings: Settings,
    state: FSMContext,
    transfer_message_groups: Sequence[MessageGroup],
) -> None:
    del state
    transfer_batches, error_text = plan_transfer_batches(transfer_message_groups, settings=settings)
    if error_text is not None:
        await message.edit_text(error_text, reply_markup=None)
        return
    if not transfer_batches:
        await message.edit_text('No clips received', reply_markup=None)
        return

    await message.edit_text('Routing...', reply_markup=None)

    async def update_transfer_progress(selection_batches: Sequence[TransferBatch]) -> None:
        await message.edit_text(
            **_transfer_progress_kwargs(selection_batches),
            reply_markup=None,
        )

    try:
        transfer_result = await _transfer_clip_batches(
            services=services,
            transfer_batches=transfer_batches,
            on_batch_transferred=update_transfer_progress,
        )
    except ClipGroupNotFoundError, UnknownClipsError:
        await message.edit_text('External clip(s)', reply_markup=None)
        return
    for clip_group, clip_sub_group in transfer_result.transfer_result.affected_sub_groups:
        if not _uses_dense_layout(clip_sub_group.scope):
            continue
        try:
            await services.clip_store.compact(
                clip_group,
                clip_sub_group,
                batch_size=_TELEGRAM_MEDIA_GROUP_LIMIT,
                require_exists=False,
            )
        except Exception:
            logger.exception(
                'post-transfer clip compaction failed for {} {}',
                clip_group,
                clip_sub_group,
            )
            raise
    await message.answer(**_transfer_summary_kwargs(transfer_result.transfer_result))


async def _clip_messages_to_clip_files(
    *,
    bot: Bot,
    messages: Sequence[Message],
) -> list[FileBytes]:
    async def to_clip_file(message: Message) -> FileBytes:
        file_id = extract_clip_file_id(message)
        if file_id is None:
            raise ValueError('Route batches must contain only clip messages')
        return FileBytes(
            data=await download_video_bytes(bot, file_id=file_id),
            extension=Extension.MP4,
        )

    # Route storage slices large route groups before calling this helper,
    # so downloads remain concurrent while each store() call stays bounded.
    # `gather()` preserves input order, which keeps the stored clip order aligned
    # with the original buffered message order.
    return list(await asyncio.gather(*(to_clip_file(message) for message in messages)))


def _pending_reconcile_clip_id_batches(
    *,
    services: Services,
    chat_id: ChatId,
) -> tuple[ClipGroup, list[list[ClipId]]]:
    return prepare_reconcile_clip_id_batches(services.chat_message_buffer.peek_grouped(chat_id))


def _has_buffered_videos(
    *,
    services: Services,
    chat_id: ChatId,
) -> bool:
    return any(message.video is not None for message in services.chat_message_buffer.peek_raw(chat_id))


def _has_buffered_clip_media(
    *,
    services: Services,
    chat_id: ChatId,
) -> bool:
    return any(extract_clip_file_id(message) is not None for message in services.chat_message_buffer.peek_raw(chat_id))


def _store_year_options(*, current_year: int, min_year: int) -> list[int]:
    return year_option_universe(current_year=current_year, min_year=min_year)


def _store_season_options(*, year: int, today: date) -> list[Season]:
    return store_allowed_seasons(year=year, today=today)


def _buffered_video_messages(message_groups: Sequence[MessageGroup]) -> list[Message]:
    # Reorder intentionally uses this same video-only flattening for both
    # peek-time validation and final flush-time execution so non-video messages
    # are ignored with identical semantics in both phases.
    return [message for message_group in message_groups for message in message_group if message.video is not None]


async def _send_reordered_video_messages(
    *,
    bot: Bot,
    chat_id: ChatId,
    messages: Sequence[Message],
) -> None:
    if not messages:
        raise ValueError('`messages` must not be empty')

    for start in range(0, len(messages), _TELEGRAM_MEDIA_GROUP_LIMIT):
        batch = messages[start : start + _TELEGRAM_MEDIA_GROUP_LIMIT]
        if len(batch) == 1:
            await bot.send_video(
                chat_id=chat_id,
                video=_video_file_id(batch[0]),
            )
            continue
        await bot.send_media_group(
            chat_id=chat_id,
            media=[InputMediaVideo(media=_video_file_id(message)) for message in batch],
        )


def _video_file_id(message: Message) -> str:
    if message.video is None:
        raise ValueError('Reorder can resend only video messages')
    return message.video.file_id


def _log_route_unsupported_codec_warning(
    *,
    error: UnsupportedVideoCodecError,
    chat_id: ChatId,
    clip_group: ClipGroup,
    sub_season: SubSeason,
    messages: Sequence[Message],
) -> None:
    route_target = (
        clip_group,
        ClipSubGroup(sub_season=sub_season, scope=Scope.SOURCE),
    )
    logger.warning(
        (
            'unsupported clip codec during route execution '
            '(codec={}, supported_codecs={}, chat_id={}, message_ids={}, file_ids={}, filenames={}, route_target={})'
        ),
        error.codec,
        error.supported_codecs,
        chat_id,
        [message.message_id for message in messages],
        [file_id for message in messages if (file_id := extract_clip_file_id(message)) is not None],
        [_route_message_filename(message) for message in messages if _route_message_filename(message) is not None],
        route_target,
    )


def _route_message_filename(message: Message) -> str | None:
    if message.video is not None:
        file_name = getattr(message.video, 'file_name', None)
        return file_name if isinstance(file_name, str) else None

    document = getattr(message, 'document', None)
    if document is None:
        return None

    file_name = getattr(document, 'file_name', None)
    return file_name if isinstance(file_name, str) else None


def _intake_action_menu_kwargs(
    *,
    services: Services,
    chat_id: ChatId,
    message_width: int,
    clip_count_override: int | None = None,
) -> dict[str, Any] | None:
    """Build the root Clips menu for any buffered clip-bearing batch.

    The root menu is intentionally broader than any single action's runtime
    validation. It should stay simple and predictable:
    - menu-time logic only decides whether the buffered batch contains clips
    - action handlers own final validation for their own media or identity
    - visible buttons are not a promise that the selected action will succeed

    This keeps menu visibility from duplicating action-specific validation,
    which would be fragile for mixed clip-media and identity-only workflows.
    """
    raw_messages = services.chat_message_buffer.peek_raw(chat_id)
    message_count = len(raw_messages)
    clip_count = clip_count_override
    has_video_clip_messages = False
    has_document_clip_messages = False
    has_route_menu_signal = False
    if clip_count is None:
        clip_count = 0
    for message in raw_messages:
        if message.video is not None:
            has_video_clip_messages = True
        if _message_has_route_menu_signal(message):
            has_route_menu_signal = True
        clip_file_id = extract_clip_file_id(message)
        if clip_file_id is not None:
            if clip_count_override is None:
                clip_count += 1
            if message.document is not None:
                has_document_clip_messages = True
    if clip_count == 0:
        return None
    buffer_version = services.chat_message_buffer.version(chat_id)
    document_only_clip_buffer = has_document_clip_messages and not has_video_clip_messages
    if has_route_menu_signal:
        route_button = _create_intake_action_button(IntakeAction.ROUTE, buffer_version=buffer_version)
        cancel_button = _create_intake_action_button(IntakeAction.CANCEL, buffer_version=buffer_version)
        reply_markup = three_row_keyboard(
            top_row=[route_button],
            middle_row=[dummy_button()],
            bottom_row=[cancel_button],
        )
    elif document_only_clip_buffer:
        action_buttons = [
            _create_intake_action_button(IntakeAction.PRODUCE, buffer_version=buffer_version),
        ]
        reply_markup = selection_keyboard(
            buttons=action_buttons,
            back_button=_create_intake_action_button(
                IntakeAction.CANCEL,
                buffer_version=buffer_version,
            ),
        )
    else:
        action_buttons = [
            _create_intake_action_button(IntakeAction.REORDER, buffer_version=buffer_version),
            _create_intake_action_button(IntakeAction.COMPACT, buffer_version=buffer_version),
            _create_intake_action_button(IntakeAction.REMOVE, buffer_version=buffer_version),
            _create_intake_action_button(IntakeAction.PRODUCE, buffer_version=buffer_version),
            _create_intake_action_button(IntakeAction.RECONCILE, buffer_version=buffer_version),
        ]
        reply_markup = selection_keyboard(
            buttons=action_buttons,
            back_button=_create_intake_action_button(
                IntakeAction.CANCEL,
                buffer_version=buffer_version,
            ),
        )
    return {
        **Text(
            create_padding_line(message_width),
            '\n',
            Text('Messages: ', Bold(str(message_count))),
            '. Select action:',
        ).as_kwargs(),
        # Root clip actions are versioned because these are destructive,
        # state-coupled callbacks that must not survive buffer changes.
        'reply_markup': reply_markup,
    }


def _message_has_route_menu_signal(message: Message) -> bool:
    if isinstance(message.text, str) and parse_route_text(message.text) is not None:
        return True
    if extract_clip_file_id(message) is None:
        return False
    if not isinstance(message.caption, str):
        return False
    return parse_route_text(message.caption) is not None


async def try_dispatch_clip_intake(
    *,
    message: Message,
    services: Services,
    settings: Settings,
) -> bool:
    """Send the root Clips menu for any buffered clip-bearing batch.

    The current buffer only needs to contain clip media for the menu to be
    shown. Individual actions perform their own authoritative validation, so
    this dispatcher does not try to predict which buttons will ultimately be
    valid for the batch.
    """
    chat_id = message.chat.id
    if not _has_buffered_clip_media(
        services=services,
        chat_id=chat_id,
    ):
        return False

    kwargs = _intake_action_menu_kwargs(
        services=services,
        chat_id=chat_id,
        message_width=settings.message_width,
    )
    if kwargs is None:
        services.chat_message_buffer.flush(chat_id)
        await message.answer(text='No clips received')
        return True

    await message.answer(**kwargs)
    return True


def _route_progress_kwargs(route_batches: Sequence[RouteBatch]) -> dict[str, Any]:
    parts: list[object] = ['Routing...']

    for route_batch in route_batches:
        parts.extend(
            [
                '\n',
                _route_progress_line(
                    selection_labels(
                        universe=route_batch.clip_group.universe,
                        year=route_batch.clip_group.year,
                        season=route_batch.clip_group.season,
                        sub_season=route_batch.sub_season,
                        scope=Scope.SOURCE,
                    )
                ),
            ]
        )

    return Text(*parts).as_kwargs()


def _transfer_progress_kwargs(transfer_batches: Sequence[TransferBatch]) -> dict[str, Any]:
    parts: list[object] = ['Routing...']

    for transfer_batch in transfer_batches:
        parts.extend(
            [
                '\n',
                _route_progress_line(
                    selection_labels(
                        universe=transfer_batch.destination_group.universe,
                        year=transfer_batch.destination_group.year,
                        season=transfer_batch.destination_group.season,
                    )
                ),
            ]
        )

    return Text(*parts).as_kwargs()


def _route_progress_line(values: Sequence[str]) -> Text:
    parts: list[object] = ['→ ']
    for index, value in enumerate(values):
        if index > 0:
            parts.append(' → ')
        parts.append(Bold(value))
    return Text(*parts)


def _create_intake_action_button(action: IntakeAction, *, buffer_version: int) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=action.title(),
        callback_data=IntakeActionCallbackData(action=action, buffer_version=buffer_version).pack(),
    )


async def _show_intake_action_menu(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    settings: Settings,
    clip_count_override: int | None = None,
    preserve_state: bool = False,
) -> None:
    if len(services.chat_message_buffer.peek_flat(message.chat.id)) == 0:
        await _invalidate_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='Invalid input',
        )
        return

    if preserve_state:
        await state.set_state(None)
    else:
        await state.clear()
    kwargs = _intake_action_menu_kwargs(
        services=services,
        chat_id=message.chat.id,
        message_width=settings.message_width,
        clip_count_override=clip_count_override,
    )
    if kwargs is None:
        await _invalidate_intake_buffer(
            message=message,
            state=state,
            services=services,
            text='No clips received',
        )
        return
    await message.edit_text(**kwargs)


async def _invalidate_intake_buffer(
    *,
    message: Message,
    state: FSMContext,
    services: Services,
    text: str,
) -> None:
    """Flush the intake buffer before rendering a non-stale invalidation.

    Intake invalidations that collapse the menu into plain text must always
    clear the buffered messages so the UI stays stateless. The only exception
    is stale-selection handling, which uses `Selection is no longer available`
    and intentionally preserves the buffer.

    This also applies to `Reconcile` pre-execution validation failures. Those
    failures are intentionally treated as hard invalidations: once the menu is
    collapsed to a generic error message, the buffered clips are discarded and
    the user must resend them.
    """
    await state.clear()
    services.chat_message_buffer.flush(message.chat.id)
    await message.edit_text(text, reply_markup=None)


def _reconcile_summary_kwargs(result: ReconcileResult) -> dict[str, Any]:
    if result.updated == 0 and result.removed == 0:
        return {'text': 'Nothing changed'}

    parts: list[object] = []

    if result.updated > 0:
        parts.extend(['Updated: ', Bold(str(result.updated))])

    if result.removed > 0:
        if parts:
            parts.append('\n')
        parts.extend(['Removed: ', Bold(str(result.removed))])

    return Text(*parts).as_kwargs()


def _transfer_summary_kwargs(result: TransferResult) -> dict[str, Any]:
    if (
        result.transferred_count == 0
        and result.already_in_destination_group_count == 0
        and result.duplicate_blocked_count == 0
    ):
        return {'text': 'Nothing changed'}

    parts: list[object] = []
    if result.transferred_count > 0:
        parts.extend(['Transferred: ', Bold(str(result.transferred_count))])
    if result.already_in_destination_group_count > 0:
        if parts:
            parts.append('\n')
        parts.extend(
            [
                'Already in destination group: ',
                Bold(str(result.already_in_destination_group_count)),
            ]
        )
    if result.duplicate_blocked_count > 0:
        if parts:
            parts.append('\n')
        parts.extend(['Duplicates exist: ', Bold(str(result.duplicate_blocked_count))])
    return Text(*parts).as_kwargs()


def _selection_flow_for_mode(mode: object) -> FlowMenuDefinition | None:
    if mode == _STORE_FLOW.mode:
        return _STORE_FLOW
    if mode == _PRODUCE_FLOW.mode:
        return _PRODUCE_FLOW
    if mode == _RECONCILE_FLOW.mode:
        return _RECONCILE_FLOW
    return None


def _is_intake_buffer_state_valid(
    *,
    data: dict[str, object],
    services: Services,
    chat_id: ChatId,
) -> bool:
    buffer_version = _buffer_version_from_state(data)
    if buffer_version is None:
        return False
    return buffer_version == services.chat_message_buffer.version(chat_id)


async def _intake_buffer_version_for_menu(
    *,
    state: FSMContext,
    buffer_version: int | None,
) -> int | None:
    if buffer_version is not None:
        return buffer_version
    return _buffer_version_from_state(await state.get_data())


async def _store_buffer_version(
    *,
    state: FSMContext,
    buffer_version: int | None,
) -> None:
    if buffer_version is None:
        return
    await state.update_data(buffer_version=buffer_version)


async def _show_intake_menu_or_stale(
    *,
    show_menu: IntakeShowMenu,
    message: Message,
    state: FSMContext,
    buffer_version: int | None = None,
    **kwargs: object,
) -> bool:
    resolved_buffer_version = await _intake_buffer_version_for_menu(
        state=state,
        buffer_version=buffer_version,
    )
    if not await show_menu(
        message=message,
        state=state,
        **kwargs,
    ):
        await handle_stale_selection(message=message, state=state)
        return False
    await _store_buffer_version(state=state, buffer_version=resolved_buffer_version)
    return True


def _buffer_version_from_state(data: dict[str, object]) -> int | None:
    buffer_version = data.get(_BUFFER_VERSION_KEY)
    if isinstance(buffer_version, int):
        return buffer_version
    return None


def _reconcile_clip_group_from_state(data: dict[str, object]) -> ClipGroup | None:
    clip_group = data.get('clip_group')
    if isinstance(clip_group, ClipGroup):
        return clip_group
    return None


def _reconcile_clip_id_batches_from_state(data: dict[str, object]) -> list[list[ClipId]] | None:
    clip_id_batches = data.get('clip_id_batches')
    if not isinstance(clip_id_batches, list):
        return None

    normalized_batches: list[list[ClipId]] = []
    for batch in clip_id_batches:
        if not isinstance(batch, list):
            return None
        normalized_batch: list[ClipId] = []
        for clip_id in batch:
            if not isinstance(clip_id, str):
                return None
            normalized_batch.append(clip_id)
        normalized_batches.append(normalized_batch)

    return normalized_batches
