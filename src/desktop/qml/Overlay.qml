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
    property bool modelLoadFailed: false
    property bool hasAvatar3D: overlayBridge.avatarModel.exists && !modelLoadFailed
    property bool connected: overlayBridge.connected || false
    property bool loading: overlayBridge.loading || false
    property bool canSend: overlayRoot.connected && !overlayRoot.loading
    property string statusBannerText: overlayRoot.loading ? "読み込み中…" : (overlayRoot.connected ? "接続済み" : "オフライン")
    property var bridgeMessages: overlayBridge.messages || []
    property var bridgeStarters: (overlayBridge.game && overlayBridge.game.starters) || []
    property var chatMessages: overlayRoot.recentMessages()
    property var starterChips: overlayRoot.starterPrompts()
    property int summaryGrowthToday: overlayBridge.growth ? overlayRoot.safeCount(overlayBridge.growth.today_points) : 0
    property int summaryGrowthStreak: overlayBridge.growth ? overlayRoot.safeCount(overlayBridge.growth.streak_days) : 0
    property int summaryOpenTasks: overlayRoot.countOpenTasks()
    property int summaryCalendarCount: overlayRoot.countUpcomingCalendar()

    onExpandedChanged: {
        if (overlayRoot.expanded) {
            composerField.forceActiveFocus()
        }
    }

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

    function recentMessages() {
        var all = overlayRoot.bridgeMessages
        var result = []
        for (var i = all.length - 1; i >= 0 && result.length < 4; i--) {
            var item = all[i]
            if (item.role === "user" || item.role === "assistant") {
                result.unshift(item)
            }
        }
        return result
    }

    function starterPrompts() {
        return overlayRoot.bridgeStarters.slice(0, 3)
    }

    function safeCount(value) {
        var n = Number(value)
        if (!isFinite(n) || n < 0) return 0
        return Math.round(n)
    }

    function countOpenTasks() {
        var list = (overlayBridge && overlayBridge.tasks) || []
        if (!list || typeof list !== "object" || typeof list.length !== "number") return 0
        var n = 0
        for (var i = 0; i < list.length; i++) {
            var item = list[i]
            if (!item || typeof item !== "object") continue
            if (!item.status || item.status === "open") n++
        }
        return n
    }

    function countUpcomingCalendar() {
        var list = (overlayBridge && overlayBridge.calendarEvents) || []
        if (!list || typeof list !== "object" || typeof list.length !== "number") return 0
        var now = new Date()
        var today = now.getFullYear() + "-" + String(now.getMonth() + 1).padStart(2, "0") + "-" + String(now.getDate()).padStart(2, "0")
        var n = 0
        for (var i = 0; i < list.length; i++) {
            var item = list[i]
            if (!item || typeof item !== "object") continue
            var start = String(item.start || "")
            if (start.length < 10 || start.charAt(4) !== "-" || start.charAt(7) !== "-") continue
            if (start.slice(0, 10) >= today) n++
        }
        return n
    }

    function growthSummaryText() {
        var streak = overlayRoot.summaryGrowthStreak
        return "+" + overlayRoot.summaryGrowthToday + "pt" + (streak > 0 ? " · " + streak + "日" : "")
    }

    function useStarter(prompt) {
        composerField.text = prompt
        composerField.forceActiveFocus()
    }

    function send() {
        if (!overlayRoot.canSend) return
        var text = composerField.text.trim()
        if (!text) return
        overlayBridge.sendMessage(text)
        composerField.clear()
        composerField.forceActiveFocus()
    }

    function collapse() {
        overlayRoot.expanded = false
    }

    Item {
        id: overlayContent
        objectName: "overlayContent"
        anchors.fill: parent
        focus: true
        Keys.onEscapePressed: overlayRoot.collapse()

        Loader {
            id: avatar3DLoader
            anchors.centerIn: parent
            width: 56
            height: 56
            active: overlayRoot.hasAvatar3D && !overlayRoot.expanded
            source: overlayRoot.hasAvatar3D ? "OverlayAvatar3D.qml" : ""
            onStatusChanged: {
                if (status === Loader.Error) {
                    overlayRoot.modelLoadFailed = true
                }
            }
            onLoaded: {
                // encodeURI: 非ASCII(ドキュメント等)・空白をパーセントエンコードしつつ :// は保持
                item.modelUrl = encodeURI("file:///" + overlayBridge.avatarModel.path.replace(/\\/g, "/"))
                item.shrink = overlayRoot.shrink
                item.dimmed = overlayRoot.shellState === "away" || overlayRoot.shrink
                item.spinning = overlayRoot.shellState === "idle" || overlayRoot.shellState === "working"
                item.loadFailed.connect(function() { overlayRoot.modelLoadFailed = true })
            }

            MouseArea {
                anchors.fill: parent
                onClicked: overlayRoot.expanded = !overlayRoot.expanded
            }
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
            visible: !overlayRoot.hasAvatar3D && !overlayRoot.expanded

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

                RowLayout {
                    objectName: "overlayStatusBanner"
                    spacing: 6
                    Rectangle {
                        implicitWidth: 8
                        implicitHeight: 8
                        radius: 4
                        color: overlayRoot.loading ? Theme.warning : (overlayRoot.connected ? Theme.green : Theme.magenta)
                    }
                    Label {
                        text: overlayRoot.statusBannerText
                        color: Theme.textMuted
                        font.family: Theme.uiFont
                        font.pixelSize: 10
                    }
                    Item {
                        Layout.fillWidth: true
                        implicitWidth: 1
                        implicitHeight: 1
                    }
                }

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
                    elide: Text.ElideRight
                    maximumLineCount: 1
                }

                RowLayout {
                    objectName: "overlayClickThroughRow"
                    Layout.fillWidth: true
                    spacing: 8

                    CheckBox {
                        id: clickThroughControl
                        objectName: "overlayClickThrough"
                        Layout.fillWidth: true
                        implicitHeight: 32
                        text: "クリックスルー"
                        checked: overlayBridge.overlayClickThrough
                        Accessible.description: "オーバーレイをクリックスルー(マウス透過)に切り替えます"
                        onClicked: {
                            var requested = clickThroughControl.checked
                            overlayBridge.setOverlayClickThrough(requested)
                            clickThroughControl.checked = overlayBridge.overlayClickThrough
                            if (overlayBridge.overlayClickThrough) {
                                overlayRoot.collapse()
                            }
                        }
                        indicator: Item {
                            implicitWidth: 0
                            implicitHeight: 0
                            visible: false
                        }
                        background: Rectangle {
                            radius: Theme.radiusSmall
                            color: clickThroughControl.checked ? Theme.accentStrong : (clickThroughControl.hovered ? Theme.panelHover : Theme.panel)
                            border.color: clickThroughControl.checked ? Theme.accent : Theme.lineSoft
                        }
                        contentItem: Label {
                            text: clickThroughControl.text
                            color: clickThroughControl.checked ? Theme.background : Theme.textMuted
                            font.family: Theme.uiFont
                            font.pixelSize: 11
                            elide: Text.ElideRight
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        Connections {
                            target: overlayBridge
                            function onOverlayClickThroughChanged() {
                                clickThroughControl.checked = overlayBridge.overlayClickThrough
                            }
                        }
                    }

                    Label {
                        objectName: "overlayClickThroughHint"
                        Layout.fillWidth: true
                        text: "Ctrl+Alt+Space で解除"
                        visible: overlayBridge.overlayClickThrough
                        color: Theme.warning
                        font.family: Theme.uiFont
                        font.pixelSize: 10
                        elide: Text.ElideRight
                        maximumLineCount: 1
                        clip: true
                    }
                }

                ColumnLayout {
                    objectName: "overlayTodaySummary"
                    Layout.fillWidth: true
                    spacing: 6
                    Accessible.description: "今日の成長ポイント・タスク・予定の件数まとめ"

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Rectangle {
                            objectName: "overlaySummaryGrowth"
                            Layout.fillWidth: true
                            implicitHeight: 40
                            radius: Theme.radiusSmall
                            color: Theme.panel
                            border.color: Theme.lineSoft
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 6
                                spacing: 2

                                Label {
                                    objectName: "overlaySummaryGrowthValue"
                                    Layout.fillWidth: true
                                    text: overlayRoot.growthSummaryText()
                                    color: Theme.accent
                                    font.family: Theme.uiFont
                                    font.bold: true
                                    font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter
                                    elide: Text.ElideRight
                                    clip: true
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: "今日の成長"
                                    color: Theme.textMuted
                                    font.family: Theme.uiFont
                                    font.pixelSize: 9
                                    horizontalAlignment: Text.AlignHCenter
                                }
                            }
                        }

                        Rectangle {
                            objectName: "overlaySummaryTasks"
                            Layout.fillWidth: true
                            implicitHeight: 40
                            radius: Theme.radiusSmall
                            color: Theme.panel
                            border.color: Theme.lineSoft
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 6
                                spacing: 2

                                Label {
                                    objectName: "overlaySummaryTasksValue"
                                    Layout.fillWidth: true
                                    text: overlayRoot.summaryOpenTasks + "件"
                                    color: Theme.text
                                    font.family: Theme.uiFont
                                    font.bold: true
                                    font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter
                                    elide: Text.ElideRight
                                    clip: true
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: "タスク"
                                    color: Theme.textMuted
                                    font.family: Theme.uiFont
                                    font.pixelSize: 9
                                    horizontalAlignment: Text.AlignHCenter
                                }
                            }
                        }

                        Rectangle {
                            objectName: "overlaySummaryCalendar"
                            Layout.fillWidth: true
                            implicitHeight: 40
                            radius: Theme.radiusSmall
                            color: Theme.panel
                            border.color: Theme.lineSoft
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 6
                                spacing: 2

                                Label {
                                    objectName: "overlaySummaryCalendarValue"
                                    Layout.fillWidth: true
                                    text: overlayRoot.summaryCalendarCount + "件"
                                    color: Theme.text
                                    font.family: Theme.uiFont
                                    font.bold: true
                                    font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter
                                    elide: Text.ElideRight
                                    clip: true
                                }
                                Label {
                                    Layout.fillWidth: true
                                    text: "予定"
                                    color: Theme.textMuted
                                    font.family: Theme.uiFont
                                    font.pixelSize: 9
                                    horizontalAlignment: Text.AlignHCenter
                                }
                            }
                        }
                    }
                }

                ColumnLayout {
                    objectName: "overlayRecentMessages"
                    Layout.fillWidth: true
                    spacing: 4
                    visible: overlayRoot.chatMessages.length > 0

                    Repeater {
                        objectName: "overlayMessageRepeater"
                        model: overlayRoot.chatMessages
                        delegate: Rectangle {
                            required property var modelData
                            objectName: "overlayMessageItem"
                            Layout.fillWidth: true
                            implicitHeight: Math.min(52, messageText.implicitHeight + 12)
                            radius: Theme.radiusSmall
                            color: modelData.role === "user" ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.16) : Theme.panel
                            border.color: modelData.role === "user" ? Theme.accent : Theme.lineSoft
                            border.width: 1
                            clip: true

                            Label {
                                id: messageText
                                anchors.fill: parent
                                anchors.margins: 6
                                text: (modelData.role === "user" ? "あなた: " : "相棒: ") + (modelData.content || "")
                                color: modelData.role === "user" ? Theme.accent : Theme.text
                                font.family: Theme.uiFont
                                font.pixelSize: 11
                                wrapMode: Text.Wrap
                                maximumLineCount: 2
                                elide: Text.ElideRight
                            }
                        }
                    }
                }

                RowLayout {
                    objectName: "overlayStarterChips"
                    Layout.fillWidth: true
                    visible: overlayRoot.starterChips.length > 0
                    spacing: 6

                    Repeater {
                        objectName: "overlayStarterRepeater"
                        model: overlayRoot.starterChips
                        delegate: Button {
                            id: starterChip
                            required property var modelData
                            objectName: "overlayStarterChip"
                            Layout.fillWidth: true
                            implicitHeight: 28
                            text: modelData.label || ""
                            onClicked: overlayRoot.useStarter(modelData.prompt || "")
                            Accessible.description: "入力欄に候補のプロンプトを設定します"
                            background: Rectangle { radius: Theme.radiusSmall; color: starterChip.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                            contentItem: Label {
                                text: starterChip.text
                                color: Theme.textMuted
                                font.family: Theme.uiFont
                                font.pixelSize: 10
                                elide: Text.ElideRight
                                clip: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    TextField {
                        id: composerField
                        objectName: "overlayComposer"
                        Layout.fillWidth: true
                        implicitHeight: 36
                        placeholderText: "メッセージを入力"
                        enabled: overlayRoot.canSend
                        selectByMouse: true
                        color: Theme.text
                        placeholderTextColor: Theme.textMuted
                        font.family: Theme.uiFont
                        font.pixelSize: 12
                        Accessible.description: "会話メッセージの入力欄"
                        onAccepted: overlayRoot.send()
                        Keys.onEscapePressed: function(event) {
                            overlayRoot.collapse()
                            event.accepted = true
                        }
                        background: Rectangle { radius: Theme.radiusSmall; color: Theme.panel; border.color: composerField.activeFocus ? Theme.accent : Theme.lineSoft }
                    }
                    Button {
                        id: sendButton
                        objectName: "overlaySendButton"
                        implicitWidth: 56
                        implicitHeight: 36
                        text: "送る"
                        enabled: overlayRoot.canSend
                        onClicked: overlayRoot.send()
                        Accessible.description: "入力を相棒へ送信します"
                        background: Rectangle { radius: Theme.radiusSmall; color: sendButton.enabled ? (sendButton.hovered ? Theme.accentStrong : Theme.accent) : Theme.panel; border.color: Theme.lineSoft }
                        contentItem: Label { text: sendButton.text; color: sendButton.enabled ? Theme.background : Theme.textMuted; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8

                    Button {
                        id: closeButton
                        text: "閉じる"
                        implicitHeight: 32
                        onClicked: overlayRoot.expanded = false
                        background: Rectangle { radius: Theme.radiusSmall; color: closeButton.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                        contentItem: Label { text: closeButton.text; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                    Button {
                        id: openMainButton
                        text: "本体を開く"
                        implicitHeight: 32
                        onClicked: overlayBridge.openMainFromOverlay()
                        background: Rectangle { radius: Theme.radiusSmall; color: openMainButton.hovered ? Theme.panelHover : Theme.accent; border.color: Theme.lineSoft }
                        contentItem: Label { text: openMainButton.text; color: Theme.background; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                    Button {
                        id: stopButton
                        text: "停止"
                        implicitHeight: 32
                        onClicked: { overlayRoot.expanded = false; overlayBridge.stopOverlayFromOverlay() }
                        Accessible.description: "オーバーレイを閉じ、PCの活動見守りセンサーも停止します"
                        background: Rectangle { radius: Theme.radiusSmall; color: stopButton.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                        contentItem: Label { text: stopButton.text; color: Theme.magenta; font.family: Theme.uiFont; font.pixelSize: 11; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                }
            }
        }
    }
}