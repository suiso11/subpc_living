pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Item {
    id: page
    property var backend
    function focusComposer() { composer.forceActiveFocus() }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 88
            radius: Theme.radius
            color: Theme.panel
            border.color: Theme.lineSoft
            RowLayout {
                anchors.fill: parent
                anchors.margins: 18
                spacing: 18
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 3
                    Label { text: "✦ いま話している相棒"; color: Theme.accent; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 12 }
                    Label {
                        text: (backend.game.rank && backend.game.rank.name) || "相棒"
                        color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 24
                    }
                }
                Label { text: Number(backend.game.points || 0).toLocaleString() + " GP"; color: Theme.text; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 18 }
                Rectangle { implicitWidth: 1; Layout.fillHeight: true; color: Theme.lineSoft }
                Label { text: backend.connected ? "● ONLINE" : "○ OFFLINE"; color: backend.connected ? Theme.green : Theme.warning; font.family: Theme.monoFont; font.pixelSize: 11 }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: Theme.radius
            color: Theme.backgroundRaised
            border.color: Theme.lineSoft
            clip: true

            ListView {
                id: messageList
                anchors.fill: parent
                anchors.margins: 14
                spacing: 12
                clip: true
                model: backend.messages
                onCountChanged: Qt.callLater(positionViewAtEnd)
                delegate: Item {
                    required property var modelData
                    width: messageList.width
                    height: bubble.implicitHeight
                    Rectangle {
                        id: bubble
                        implicitHeight: messageText.implicitHeight + 28
                        width: Math.min(messageList.width * 0.78, Math.max(200, messageText.implicitWidth + 30))
                        x: modelData.role === "user" ? messageList.width - width : 0
                        radius: Theme.radius
                        color: modelData.role === "user" ? Theme.accent : Theme.panelStrong
                        border.color: modelData.role === "user" ? Theme.accent : Theme.line
                        Label {
                            id: messageText
                            anchors.fill: parent
                            anchors.margins: 14
                            text: modelData.content || ""
                            color: modelData.role === "user" ? Theme.background : Theme.text
                            font.family: Theme.uiFont
                            font.pixelSize: 15
                            wrapMode: Text.Wrap
                        }
                    }
                }
                Label {
                    anchors.centerIn: parent
                    visible: messageList.count === 0
                    text: "ここから話の続きを始められます"
                    color: Theme.textMuted
                    font.family: Theme.uiFont
                    font.pixelSize: 15
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            TextArea {
                id: composer
                Layout.fillWidth: true
                implicitHeight: Math.min(104, Math.max(48, contentHeight + 22))
                placeholderText: "なんでも話して"
                color: Theme.text
                placeholderTextColor: Theme.textMuted
                font.family: Theme.uiFont
                font.pixelSize: 15
                wrapMode: TextEdit.Wrap
                selectByMouse: true
                background: Rectangle { radius: Theme.radiusSmall; color: Theme.panel; border.color: composer.activeFocus ? Theme.accent : Theme.lineSoft }
                Keys.onReturnPressed: function(event) {
                    if (event.modifiers & Qt.ShiftModifier) {
                        event.accepted = false
                    } else {
                        page.send()
                        event.accepted = true
                    }
                }
            }
            Button {
                id: micButton
                implicitWidth: 52; implicitHeight: 48
                text: backend.recording ? "■" : "●"
                ToolTip.visible: hovered
                ToolTip.text: "押している間だけ話す"
                onPressed: backend.startRecording()
                onReleased: backend.stopRecording()
                onCanceled: if (backend.recording) backend.stopRecording()
                background: Rectangle { radius: Theme.radiusSmall; color: backend.recording ? Theme.magenta : micButton.hovered ? Theme.panelHover : Theme.panel; border.color: backend.recording ? Theme.magenta : Theme.line }
                contentItem: Label { text: micButton.text; color: backend.recording ? "white" : Theme.accent; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font.pixelSize: 15 }
            }
            Button {
                id: sendButton
                implicitWidth: 72; implicitHeight: 48
                text: "送る"
                onClicked: page.send()
                background: Rectangle { radius: Theme.radiusSmall; color: sendButton.hovered ? Theme.accentStrong : Theme.accent }
                contentItem: Label { text: sendButton.text; color: Theme.background; font.family: Theme.uiFont; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
            }
        }
    }

    function send() {
        const text = composer.text.trim()
        if (!text) return
        backend.sendMessage(text)
        composer.clear()
    }
}
