pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import "."

Window {
    id: overlayRoot
    objectName: "overlayRoot"
    visible: overlayBridge.overlayEnabled && overlayBridge.overlayShell.visible
    width: expanded ? 260 : 60
    height: expanded ? hudColumn.implicitHeight + 32 : 68
    color: "transparent"
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool

    x: Screen.desktopAvailableWidth - width - 12
    y: Math.round((Screen.desktopAvailableHeight - height) / 2)

    property bool expanded: false
    property var shell: overlayBridge.overlayShell || {}
    property string shellState: shell.shell_state || "idle"
    property bool shrink: shell.shrink || false

    function stateColor() {
        switch (overlayRoot.shellState) {
        case "idle": return Theme.textMuted
        case "working": return Theme.accent
        case "conversing": return Theme.green
        case "away": return Theme.textMuted
        case "schedule_near": return Theme.warning
        case "error": return Theme.magenta
        default: return Theme.textMuted
        }
    }

    function stateLabel() {
        switch (overlayRoot.shellState) {
        case "idle": return "待機中"
        case "working": return "作業中"
        case "conversing": return "会話中"
        case "away": return "離席中"
        case "schedule_near": return "予定が近づいています"
        case "error": return "エラー"
        default: return "待機中"
        }
    }

    function formatTime(ts) {
        if (!ts) return "--:--"
        var d = new Date(ts * 1000)
        var h = String(d.getHours()).padStart(2, "0")
        var m = String(d.getMinutes()).padStart(2, "0")
        return h + ":" + m
    }

    Rectangle {
        objectName: "overlayAvatar"
        anchors.centerIn: parent
        width: 56
        height: 56
        radius: 28
        color: "transparent"
        border.color: overlayRoot.stateColor()
        border.width: 2
        opacity: overlayRoot.shellState === "away" ? 0.5 : 1.0

        Rectangle {
            anchors.centerIn: parent
            width: 12
            height: 12
            radius: 6
            color: overlayRoot.stateColor()
        }

        SequentialAnimation on scale {
            running: !overlayRoot.shrink && !overlayRoot.expanded
            loops: Animation.Infinite
            NumberAnimation { from: 1.0; to: 1.05; duration: 1500; easing.type: Easing.InOutSine }
            NumberAnimation { from: 1.05; to: 1.0; duration: 1500; easing.type: Easing.InOutSine }
        }

        MouseArea {
            anchors.fill: parent
            onClicked: overlayRoot.expanded = !overlayRoot.expanded
        }
    }

    Rectangle {
        id: hudPanel
        objectName: "overlayHud"
        visible: overlayRoot.expanded
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: hudColumn.implicitHeight + 32
        radius: Theme.radius
        color: Qt.rgba(Theme.panel.r, Theme.panel.g, Theme.panel.b, 0.92)
        border.color: Theme.lineSoft
        border.width: 1

        ColumnLayout {
            id: hudColumn
            anchors.fill: parent
            anchors.margins: 16
            spacing: 10

            Label {
                objectName: "overlayStateLabel"
                text: overlayRoot.stateLabel()
                color: Theme.text
                font.family: Theme.uiFont
                font.bold: true
                font.pixelSize: 14
            }

            Label {
                objectName: "overlayProvenance"
                text: {
                    var prov = overlayRoot.shell.provenance
                    if (!prov) return ""
                    return "出所: " + (prov.source_label || "?") + " · 取得: " + overlayRoot.formatTime(prov.fetched_at) + " · 保存: なし"
                }
                color: Theme.textMuted
                font.family: Theme.monoFont
                font.pixelSize: 10
                Layout.fillWidth: true
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Button {
                    text: "閉じる"
                    implicitHeight: 32
                    onClicked: overlayRoot.expanded = false
                    background: Rectangle { radius: Theme.radiusSmall; color: parent.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                    contentItem: Label { text: parent.text; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
                Button {
                    text: "本体を開く"
                    implicitHeight: 32
                    onClicked: overlayBridge.openMainFromOverlay()
                    background: Rectangle { radius: Theme.radiusSmall; color: parent.hovered ? Theme.panelHover : Theme.accent; border.color: Theme.lineSoft }
                    contentItem: Label { text: parent.text; color: Theme.background; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
                Button {
                    text: "停止"
                    implicitHeight: 32
                    onClicked: { overlayRoot.expanded = false; overlayBridge.stopOverlayFromOverlay() }
                    background: Rectangle { radius: Theme.radiusSmall; color: parent.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                    contentItem: Label { text: parent.text; color: Theme.magenta; font.family: Theme.uiFont; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
            }
        }
    }
}
