"""ContextBuilder。

ContextPolicy.select を通した ContextBlock だけを描画し、既存互換の
role/content dict 列を組み立てる。会話本文の JSON 化・復号や Policy 外の
side-channel は使わない。入力の blocks と base system は変更しない。

描画順契約:
- str content の block は Policy 選択順に base system へ直接連結される
  (system 本文に吸収される)。
- 構造化 content (ContextMessage列) の block は LLM 契約上 system message の
  後ろへ dict 列として配置される。
- したがって、Policy の priority / filter が異種 content 間の選択に使われても、
  structured message を system message より前に置いたり、異種 content 間の
  priority で message 位置を交差させたりはしない。
- 同種 content 内の並びは Policy の選択順を保つ。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.context.contracts import ContextBlock
from src.context.policy import ContextPolicy
from src.llm.routing.contracts import PrivacyMode


class ContextBuilder:
    """base system へ str content を連結し、構造化メッセージを dict 列に描画する。"""

    def __init__(self, base_system: str = "") -> None:
        self._base_system = base_system

    def build_messages(
        self,
        blocks: Sequence[ContextBlock] | Iterable[ContextBlock],
        privacy: PrivacyMode = "local_only",
        target_local: bool = True,
    ) -> list[dict]:
        """Policy で選択された block だけを描画してメッセージ列を返す。

        - str content の block は base system へ選択順に直接連結する
        - ContextMessage tuple の block は system message の後へ
          {"role": ..., "content": ...} dict として追加する
        - 異種 content 間の priority は message 位置を交差させない。
          structured block は常に system message の後にのみ配置される。
        - blocks と base system は変更しない
        """
        selected = ContextPolicy.select(blocks, privacy=privacy, target_local=target_local)
        system_content = self._base_system
        role_messages: list[dict] = []
        for block in selected:
            if isinstance(block.content, str):
                system_content = system_content + block.content
            else:
                role_messages.extend(
                    {"role": message.role, "content": message.content}
                    for message in block.content
                )

        messages: list[dict] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.extend(role_messages)
        return messages
