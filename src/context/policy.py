"""決定的ContextPolicy。

ContextBlock の機密度・local_only・送信先・優先度という metadata だけを見て
送信可否と順序を決める。会話本文の文字列判定は行わない。
History は ContextBlock 化の前段階にあり、このポリシーの入力対象外である。

このポリシーが検査するのはprivacy metadataのみで、allow_cloud によるcloud送信の
最終承認は Router / AssistantService の責務である。ContextPolicy 単独を cloud
送信の最終ゲートとみなしてはならない。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.context.contracts import ContextBlock, TASKS_SOURCE, VALID_PRIVACY_MODES
from src.llm.routing.contracts import PrivacyMode


class ContextPolicyError(ValueError):
    """許可されないprivacyが指定された。"""


class ContextPolicy:
    """ContextBlockの機密度・送信先・優先度だけを見る決定的ポリシー。"""

    @staticmethod
    def select(
        blocks: Sequence[ContextBlock] | Iterable[ContextBlock],
        privacy: PrivacyMode = "local_only",
        target_local: bool = True,
    ) -> tuple[ContextBlock, ...]:
        """送信候補を返す。入力の list や block 自体は変更しない。

        privacy は送信先への許可であり、local target の sensitivity 除外ではない。
        未知の privacy は target_local に関係なく ContextPolicyError で拒否する。

        - target_local=True なら全 sensitivity を許可する。
        - target_local=False は privacy=cloud_allowed のときだけ検査を通過し、
          local_only / local_preferred を ContextPolicyError で拒否する。
          通過できるのは public かつ local_only=False のブロックだけ。
          この検査はmetadata判定に過ぎず、allow_cloud によるcloud送信の最終承認は
          Router / AssistantService の責務である。
        - 順序は priority 昇順の stable sort。source=tasks は priority に関係なく最後。
        """
        blocks = tuple(blocks)
        if privacy not in VALID_PRIVACY_MODES:
            raise ContextPolicyError(f"unknown privacy: {privacy!r}")
        if not target_local and privacy != "cloud_allowed":
            raise ContextPolicyError(
                f"non-local target requires privacy=cloud_allowed, got {privacy!r}"
            )
        if target_local:
            allowed = blocks
        else:
            allowed = tuple(
                block
                for block in blocks
                if block.sensitivity == "public" and not block.local_only
            )
        return tuple(
            sorted(allowed, key=lambda block: (block.source == TASKS_SOURCE, block.priority))
        )