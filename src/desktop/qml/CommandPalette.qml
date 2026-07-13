pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Popup {
    id: palette
    property var hostWindow
    property var taskPage
    property var chatPage
    modal: true
    focus: true
    width: Math.min(620, hostWindow.width - 48)
    height: Math.min(540, hostWindow.height - 80)
    x: Math.round((hostWindow.width - width) / 2)
    y: 76
    padding: 0
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property var commands: [
        { id: "talk", label: "話す", detail: "会話に戻る", key: "Alt+1" },
        { id: "tasks", label: "やること", detail: "優先順位と最初の一歩を見る", key: "Alt+2" },
        { id: "logs", label: "記録", detail: "会話とシステムの動きを振り返る", key: "Alt+3" },
        { id: "game", label: "実績", detail: "相棒との積み重ねを見る", key: "Alt+4" },
        { id: "add", label: "タスクを追加", detail: "やることを1行で登録", key: "N" },
        { id: "refresh", label: "いまの画面を読み直す", detail: "バックエンドから最新状態を取得", key: "R" },
        { id: "settings", label: "接続と常駐の設定", detail: "サーバー・自動起動・音声", key: "," }
    ]

    function openPalette() {
        query.text = ""
        selectedIndex = 0
        open()
        query.forceActiveFocus()
    }

    property int selectedIndex: 0
    function filtered() {
        const needle = query.text.trim().toLowerCase()
        if (!needle) return commands
        return commands.filter(c => (c.label + " " + c.detail).toLowerCase().indexOf(needle) >= 0)
    }
    function execute(commandId) {
        close()
        if (commandId === "talk") hostWindow.currentPage = 0
        else if (commandId === "tasks") hostWindow.currentPage = 1
        else if (commandId === "logs") hostWindow.currentPage = 2
        else if (commandId === "game") hostWindow.currentPage = 3
        else if (commandId === "add") {
            hostWindow.currentPage = 1
            taskPage.focusAdd()
        } else if (commandId === "settings") hostWindow.openSettings()
        else if (commandId === "refresh") hostWindow.refreshCurrentPage()
    }

    background: Rectangle {
        radius: Theme.radiusLarge
        color: Theme.backgroundRaised
        border.color: Theme.line
        border.width: 1
    }
    Overlay.modal: Rectangle { color: "#990B0605" }

    contentItem: ColumnLayout {
        spacing: 0
        RowLayout {
            Layout.fillWidth: true
            Layout.margins: 14
            spacing: 10
            Label { text: "⌘"; color: Theme.accent; font.pixelSize: 20 }
            TextField {
                id: query
                Layout.fillWidth: true
                placeholderText: "移動または操作を検索"
                color: Theme.text
                placeholderTextColor: Theme.textMuted
                font.family: Theme.uiFont
                font.pixelSize: 16
                selectByMouse: true
                background: Item {}
                onTextChanged: palette.selectedIndex = 0
                Keys.onDownPressed: palette.selectedIndex = Math.min(resultList.count - 1, palette.selectedIndex + 1)
                Keys.onUpPressed: palette.selectedIndex = Math.max(0, palette.selectedIndex - 1)
                Keys.onReturnPressed: {
                    const values = palette.filtered()
                    if (values.length) palette.execute(values[palette.selectedIndex].id)
                }
            }
            Label { text: "ESC"; color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 11 }
        }
        Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: Theme.lineSoft }
        ListView {
            id: resultList
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 10
            spacing: 4
            clip: true
            model: palette.filtered()
            currentIndex: palette.selectedIndex
            delegate: Rectangle {
                required property var modelData
                required property int index
                width: resultList.width
                height: 58
                radius: Theme.radiusSmall
                color: ListView.isCurrentItem ? Theme.accent : hover.hovered ? Theme.panelHover : "transparent"
                HoverHandler { id: hover; onHoveredChanged: if (hovered) palette.selectedIndex = index }
                TapHandler { onTapped: palette.execute(modelData.id) }
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 14
                    anchors.rightMargin: 14
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 1
                        Label {
                            text: modelData.label
                            color: ListView.isCurrentItem ? Theme.background : Theme.text
                            font.family: Theme.uiFont; font.bold: true; font.pixelSize: 14
                        }
                        Label {
                            text: modelData.detail
                            color: ListView.isCurrentItem ? "#7C4139" : Theme.textMuted
                            font.family: Theme.uiFont; font.pixelSize: 11
                        }
                    }
                    Label {
                        text: modelData.key
                        color: ListView.isCurrentItem ? Theme.background : Theme.textMuted
                        font.family: Theme.monoFont; font.pixelSize: 11
                    }
                }
            }
        }
        Label {
            Layout.leftMargin: 22
            Layout.bottomMargin: 14
            text: "↑↓ 選ぶ　Enter 実行"
            color: Theme.textMuted
            font.family: Theme.uiFont
            font.pixelSize: 11
        }
    }
}
