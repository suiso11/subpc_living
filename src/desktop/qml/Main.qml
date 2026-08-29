pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import QtQuick.Window
import "."

ApplicationWindow {
    id: root
    width: 1220
    height: 800
    minimumWidth: 820
    minimumHeight: 560
    visible: true
    title: "SUBPC BUDDY"
    color: "transparent"
    flags: Qt.Window | Qt.FramelessWindowHint
    property int currentPage: 0
    property bool closeToTray: false

    function openSettings() {
        serverInput.text = bridge.serverUrl
        settingsPopup.open()
        serverInput.forceActiveFocus()
    }
    function refreshCurrentPage() {
        if (currentPage === 0) { bridge.resumeChat(); bridge.loadGame(); bridge.loadGrowth() }
        else if (currentPage === 1) { bridge.loadTasks(); if (taskPage.mode === 1) taskPage.openCalendar() }
        else if (currentPage === 2) logsPage.activate()
        else bridge.loadGame()
    }
    function companionText() {
        var cs = bridge.companionState
        if (!cs.enabled) return ""
        var s = cs.state
        if (!s) return "相棒: 起動中"
        if (s.activity_mode === "focused") return "相棒: 集中"
        if (s.activity_mode === "idle") return "相棒: 待機"
        if (s.activity_mode === "away" || !s.present) return "相棒: 離席"
        return "相棒: 観察中"
    }
    onCurrentPageChanged: {
        if (currentPage === 0) { bridge.loadGame(); bridge.loadGrowth() }
        else if (currentPage === 1) bridge.loadTasks()
        else if (currentPage === 2) logsPage.activate()
        else if (currentPage === 3) bridge.loadGame()
    }
    onClosing: function(close) {
        if (closeToTray) {
            close.accepted = false
            root.hide()
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Qt.platform.os === "windows" ? Qt.rgba(18 / 255, 11 / 255, 10 / 255, 0.94) : Theme.background
        Rectangle {
            width: 520; height: 520; radius: 260
            x: parent.width - 300; y: -210
            color: "transparent"; border.color: "#2A1714"; border.width: 1
        }
        Rectangle {
            width: 620; height: 620; radius: 310
            x: parent.width - 180; y: 420
            color: "transparent"; border.color: "#20120F"; border.width: 1
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 24
        anchors.rightMargin: 24
        anchors.bottomMargin: 22
        spacing: 12

        Item {
            Layout.fillWidth: true
            implicitHeight: 74
            MouseArea {
                anchors.fill: parent
                anchors.rightMargin: 158
                onPressed: root.startSystemMove()
                onDoubleClicked: root.visibility = root.visibility === Window.Maximized ? Window.Windowed : Window.Maximized
            }

            Rectangle {
                id: brandPill
                objectName: "brandPill"
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                visible: root.width >= 940
                implicitWidth: brandRow.implicitWidth + 28
                implicitHeight: 44
                radius: 14
                color: Theme.backgroundRaised
                border.color: Theme.lineSoft
                RowLayout {
                    id: brandRow
                    anchors.centerIn: parent
                    spacing: 10
                    Rectangle { implicitWidth: 8; implicitHeight: 8; radius: 4; color: bridge.connected ? Theme.accent : Theme.warning }
                    Label { text: "SUBPC BUDDY"; color: Theme.text; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 12; font.letterSpacing: 0.8 }
                }
            }

            Rectangle {
                id: navigationPill
                objectName: "navigationPill"
                anchors.centerIn: parent
                implicitWidth: navRow.implicitWidth + 8
                implicitHeight: 48
                radius: 15
                color: Theme.backgroundRaised
                border.color: Theme.lineSoft
                RowLayout {
                    id: navRow
                    anchors.centerIn: parent
                    spacing: 2
                    Repeater {
                        model: ["話す", "やること", "記録", "実績"]
                        delegate: Button {
                            id: navButton
                            required property string modelData
                            required property int index
                            text: modelData
                            implicitWidth: root.width >= 1000 ? 88 : 78
                            implicitHeight: 40
                            onClicked: root.currentPage = navButton.index
                            background: Rectangle {
                                radius: 11
                                color: root.currentPage === navButton.index ? Theme.accent : navButton.hovered ? Theme.panelHover : "transparent"
                                Behavior on color { ColorAnimation { duration: Theme.motionFast } }
                            }
                            contentItem: Label {
                                text: navButton.text
                                color: root.currentPage === navButton.index ? Theme.background : Theme.textMuted
                                font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13
                                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                            }
                        }
                    }
                }
            }

            RowLayout {
                id: windowControls
                objectName: "windowControls"
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                spacing: 6
                Rectangle {
                    id: commandHint
                    objectName: "commandHint"
                    visible: root.width >= 1080
                    implicitWidth: shortcutRow.implicitWidth + 20
                    implicitHeight: 40
                    radius: 12
                    border.color: Theme.lineSoft
                    RowLayout {
                        id: shortcutRow
                        anchors.centerIn: parent
                        spacing: 7
                        Label { text: "Ctrl K"; color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 10 }
                        Label { text: "+"; color: Theme.textMuted; font.pixelSize: 15 }
                        Label { text: "•••"; color: Theme.text; font.pixelSize: 13 }
                    }
                    TapHandler { onTapped: commandPalette.openPalette() }
                    HoverHandler { id: commandHover }
                    color: commandHover.hovered ? Theme.panelHover : Theme.backgroundRaised
                }
                Button {
                    id: minimizeButton
                    implicitWidth: 34; implicitHeight: 34; text: "—"
                    onClicked: root.showMinimized()
                    background: Rectangle { radius: 9; color: minimizeButton.hovered ? Theme.panelHover : "transparent" }
                    contentItem: Label { text: minimizeButton.text; color: Theme.textMuted; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
                Button {
                    id: maximizeButton
                    implicitWidth: 34; implicitHeight: 34; text: root.visibility === Window.Maximized ? "❐" : "□"
                    onClicked: root.visibility = root.visibility === Window.Maximized ? Window.Windowed : Window.Maximized
                    background: Rectangle { radius: 9; color: maximizeButton.hovered ? Theme.panelHover : "transparent" }
                    contentItem: Label { text: maximizeButton.text; color: Theme.textMuted; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
                Button {
                    id: closeButton
                    implicitWidth: 34; implicitHeight: 34; text: "×"
                    onClicked: root.close()
                    background: Rectangle { radius: 9; color: closeButton.hovered ? Theme.warning : "transparent" }
                    contentItem: Label { text: closeButton.text; color: closeButton.hovered ? Theme.background : Theme.textMuted; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font.pixelSize: 16 }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Label {
                text: ["会話を続ける", "次の一歩を決める", "積み重ねを振り返る", "相棒との歩み"][root.currentPage]
                color: Theme.text
                font.family: Theme.uiFont; font.bold: true; font.pixelSize: 29
            }
            Label {
                Layout.fillWidth: true
                text: bridge.statusText
                color: bridge.connected ? Theme.green : Theme.warning
                font.family: Theme.monoFont; font.pixelSize: 10
                elide: Text.ElideRight
            }
            Rectangle {
                id: companionPill
                objectName: "companionPill"
                visible: bridge.companionState.enabled
                implicitWidth: companionRow.implicitWidth + 16
                implicitHeight: 26
                radius: Theme.radiusSmall
                color: Theme.backgroundRaised
                border.color: Theme.lineSoft
                RowLayout {
                    id: companionRow
                    anchors.centerIn: parent
                    spacing: 6
                    Rectangle { implicitWidth: 6; implicitHeight: 6; radius: 3; color: bridge.companionState.running ? Theme.green : Theme.warning }
                    Label {
                        text: companionText()
                        color: Theme.textMuted
                        font.family: Theme.monoFont
                        font.pixelSize: 10
                    }
                }
            }
            BusyIndicator { running: bridge.loading; visible: running; implicitWidth: 28; implicitHeight: 28 }
            Button {
                id: refreshButton
                text: "↻ 読み直す"
                onClicked: root.refreshCurrentPage()
                background: Rectangle { radius: Theme.radiusSmall; color: refreshButton.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                contentItem: Label { text: refreshButton.text; color: Theme.text; font.family: Theme.uiFont; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            }
            Button {
                id: settingsButton
                text: "設定"
                onClicked: root.openSettings()
                background: Rectangle { radius: Theme.radiusSmall; color: settingsButton.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft }
                contentItem: Label { text: settingsButton.text; color: Theme.textMuted; font.family: Theme.uiFont; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: root.currentPage
            ChatPage { id: chatPage; objectName: "chatPage"; backend: bridge }
            TasksPage { id: taskPage; objectName: "taskPage"; backend: bridge }
            LogsPage { id: logsPage; objectName: "logsPage"; backend: bridge }
            AchievementsPage { id: achievementsPage; objectName: "achievementsPage"; backend: bridge }
        }
    }

    CommandPalette {
        id: commandPalette
        objectName: "commandPalette"
        hostWindow: root
        taskPage: taskPage
        chatPage: chatPage
    }
    Toast {
        id: toast
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        z: 1000
    }
    Connections {
        target: bridge
        function onToast(title, detail) { toast.show(title, detail) }
    }

    Popup {
        id: settingsPopup
        modal: true
        focus: true
        width: Math.min(580, root.width - 60)
        height: settingsColumn.implicitHeight + 42
        x: Math.round((root.width - width) / 2)
        y: Math.round((root.height - height) / 2)
        padding: 20
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        background: Rectangle { radius: Theme.radiusLarge; color: Theme.backgroundRaised; border.color: Theme.line }
        QQC2.Overlay.modal: Rectangle { color: "#990B0605" }
        contentItem: ColumnLayout {
            id: settingsColumn
            spacing: 12
            Label { text: "接続と常駐"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 22 }
            Label { text: "AIを動かしているsubpc-webのURL"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11 }
            TextField {
                id: serverInput
                Layout.fillWidth: true
                placeholderText: "http://127.0.0.1:8000"
                color: Theme.text; font.family: Theme.monoFont
                selectByMouse: true
                background: Rectangle { radius: Theme.radiusSmall; color: Theme.panel; border.color: serverInput.activeFocus ? Theme.accent : Theme.lineSoft }
            }
            RowLayout {
                Layout.fillWidth: true
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 2
                    Label { text: "Windowsログイン時に起動"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13 }
                    Label { text: "起動後はタスクトレイに常駐します"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                }
                Switch { checked: bridge.autostartEnabled; onToggled: bridge.setAutostart(checked) }
            }
            RowLayout {
                Layout.fillWidth: true
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 2
                    Label { text: "返答を声で読む"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13 }
                    Label { text: "バックエンドのTTS音声をWindowsで再生"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                }
                Switch { checked: bridge.ttsEnabled; onToggled: bridge.setTtsEnabled(checked) }
            }
            RowLayout {
                Layout.fillWidth: true
                ColumnLayout {
                    Layout.fillWidth: true; spacing: 2
                    Label { text: "読み上げる声"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13 }
                    Label { text: "返答の自動読み上げと再生ボタンに使います"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                }
                BuddyComboBox {
                    id: voiceSelector
                    Layout.preferredWidth: 220
                    model: (bridge.ttsVoices || []).map(voice => voice.label)
                    currentIndex: Math.max(0, (bridge.ttsVoices || []).findIndex(voice => voice.id === bridge.ttsVoice))
                    enabled: count > 0
                    onActivated: function(index) { bridge.setTtsVoice(bridge.ttsVoices[index].id) }
                }
            }
            Rectangle { Layout.fillWidth: true; implicitHeight: 1; color: Theme.lineSoft }
            Label { text: "グローバル呼び出し  Ctrl + Alt + Space"; color: Theme.accent; font.family: Theme.monoFont; font.pixelSize: 11 }
            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }
                BuddyButton { text: "閉じる"; onClicked: settingsPopup.close() }
                BuddyButton {
                    id: saveSettings
                    text: "接続する"
                    accent: true
                    enabled: serverInput.text.trim().length > 0 && !bridge.loading
                    onClicked: { bridge.setServerUrl(serverInput.text); settingsPopup.close() }
                    background: Rectangle { radius: Theme.radiusSmall; color: saveSettings.enabled ? Theme.accent : Theme.lineSoft }
                    contentItem: Label { text: saveSettings.text; color: Theme.background; font.family: Theme.uiFont; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                }
            }
        }
    }

    Shortcut { sequence: "Ctrl+K"; onActivated: commandPalette.openPalette() }
    Shortcut { sequence: "Ctrl+N"; onActivated: { root.currentPage = 0; bridge.newSession() } }
    Shortcut { sequence: "Alt+1"; onActivated: root.currentPage = 0 }
    Shortcut { sequence: "Alt+2"; onActivated: root.currentPage = 1 }
    Shortcut { sequence: "Alt+3"; onActivated: root.currentPage = 2 }
    Shortcut { sequence: "Alt+4"; onActivated: root.currentPage = 3 }
    Shortcut { sequence: "Ctrl+,"; onActivated: root.openSettings() }

    // Frameless windows still keep the native Windows resize gesture.
    MouseArea {
        z: 2000; anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom
        width: 5; cursorShape: Qt.SizeHorCursor
        onPressed: root.startSystemResize(Qt.LeftEdge)
    }
    MouseArea {
        z: 2000; anchors.right: parent.right; anchors.top: parent.top; anchors.bottom: parent.bottom
        width: 5; cursorShape: Qt.SizeHorCursor
        onPressed: root.startSystemResize(Qt.RightEdge)
    }
    MouseArea {
        z: 2000; anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right
        height: 5; cursorShape: Qt.SizeVerCursor
        onPressed: root.startSystemResize(Qt.TopEdge)
    }
    MouseArea {
        z: 2000; anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right
        height: 5; cursorShape: Qt.SizeVerCursor
        onPressed: root.startSystemResize(Qt.BottomEdge)
    }
}
