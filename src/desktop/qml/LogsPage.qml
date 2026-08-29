pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import "."

Item {
    id: page
    property var backend
    property int mode: 0
    function activate() {
        if (mode === 0) backend.loadHistories()
        else if (mode === 1) backend.loadLogs(unit.values[unit.currentIndex], Number(lines.currentText))
        else backend.loadLogFiles()
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            Label { text: "記録をみる"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 27 }
            Label { Layout.fillWidth: true; text: "会話やシステムの動きを、ここから確認できます。"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 12 }
            Rectangle {
                implicitWidth: tabs.implicitWidth + 8
                implicitHeight: 42
                radius: Theme.radiusSmall
                color: Theme.panel
                border.color: Theme.lineSoft
                RowLayout {
                    id: tabs
                    anchors.centerIn: parent
                    spacing: 4
                    Repeater {
                        model: ["会話", "システム", "アプリ"]
                        delegate: Button {
                            id: tabButton
                            required property string modelData
                            required property int index
                            text: modelData
                            implicitWidth: 92; implicitHeight: 34
                            onClicked: { page.mode = index; page.activate() }
                            background: Rectangle { radius: 8; color: page.mode === index ? Theme.accent : "transparent" }
                            contentItem: Label { text: tabButton.text; color: page.mode === tabButton.index ? Theme.background : Theme.textMuted; font.family: Theme.uiFont; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                    }
                }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal
            visible: page.mode === 0
            Rectangle {
                SplitView.preferredWidth: 330
                SplitView.minimumWidth: 250
                color: Theme.backgroundRaised
                radius: Theme.radius
                border.color: Theme.lineSoft
                ListView {
                    id: historyList
                    anchors.fill: parent
                    anchors.margins: 9
                    spacing: 6
                    clip: true
                    model: backend.histories
                    delegate: Rectangle {
                        id: historyRow
                        required property var modelData
                        width: historyList.width
                        height: 70
                        radius: Theme.radiusSmall
                        color: historyHover.hovered ? Theme.panelHover : Theme.panel
                        border.color: Theme.lineSoft
                        HoverHandler { id: historyHover }
                        TapHandler { onTapped: backend.loadHistory(historyRow.modelData.file) }
                        ColumnLayout {
                            anchors.fill: parent; anchors.margins: 12; anchors.rightMargin: 52; spacing: 3
                            Label { Layout.fillWidth: true; text: historyRow.modelData.preview || "（発言なし）"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13; elide: Text.ElideRight }
                            Label { text: (historyRow.modelData.turn_count || 0) + "往復  ·  " + (historyRow.modelData.saved_at || ""); color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 10 }
                        }
                        Button {
                            id: deleteHistoryButton
                            anchors.right: parent.right; anchors.verticalCenter: parent.verticalCenter; anchors.rightMargin: 9
                            implicitWidth: 32; implicitHeight: 32; text: "×"
                            ToolTip.visible: hovered; ToolTip.text: "この記録を削除"
                            onClicked: { deleteDialog.fileName = historyRow.modelData.file; deleteDialog.open() }
                            background: Rectangle { radius: 16; color: deleteHistoryButton.hovered ? Theme.warning : "transparent"; border.color: Theme.lineSoft }
                            contentItem: Label { text: deleteHistoryButton.text; color: deleteHistoryButton.hovered ? Theme.background : Theme.textMuted; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                    }
                    Label { anchors.centerIn: parent; visible: historyList.count === 0; text: "会話の記録はまだありません"; color: Theme.textMuted; font.family: Theme.uiFont }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
            Rectangle {
                SplitView.fillWidth: true
                color: Theme.backgroundRaised
                radius: Theme.radius
                border.color: Theme.lineSoft
                ListView {
                    id: detailList
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 9
                    clip: true
                    model: backend.historyMessages
                    delegate: Rectangle {
                        required property var modelData
                        width: detailList.width
                        height: detailText.implicitHeight + 24
                        radius: Theme.radiusSmall
                        color: modelData.role === "user" ? Theme.panelStrong : Theme.panel
                        border.color: Theme.lineSoft
                        Label {
                            id: detailText
                            anchors.fill: parent; anchors.margins: 12
                            text: modelData.content || ""
                            color: Theme.text
                            font.family: Theme.uiFont; font.pixelSize: 13
                            wrapMode: Text.Wrap
                        }
                    }
                    Label { anchors.centerIn: parent; visible: detailList.count === 0; text: "左の会話を選ぶと、ここに内容が表示されます"; color: Theme.textMuted; font.family: Theme.uiFont }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: page.mode === 1
            radius: Theme.radius
            color: Theme.backgroundRaised
            border.color: Theme.lineSoft
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 10
                RowLayout {
                    Layout.fillWidth: true
                    BuddyComboBox {
                        id: unit
                        property var values: ["subpc-web", "subpc-discord", "subpc-sbv2-tts", "subpc-gpu-powersave"]
                        model: values
                        Layout.preferredWidth: 210
                    }
                    BuddyComboBox { id: lines; model: ["100", "200", "500", "1000"]; currentIndex: 1; Layout.preferredWidth: 100 }
                    BuddyButton { text: "読み直す"; onClicked: backend.loadLogs(unit.values[unit.currentIndex], Number(lines.currentText)) }
                    Item { Layout.fillWidth: true }
                    Label { text: backend.statusText; color: backend.connected ? Theme.green : Theme.warning; font.family: Theme.monoFont; font.pixelSize: 10 }
                }
                TextArea {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    text: backend.logs
                    readOnly: true
                    selectByMouse: true
                    wrapMode: TextEdit.NoWrap
                    color: Theme.textMuted
                    font.family: Theme.monoFont
                    font.pixelSize: 11
                    background: Rectangle { radius: Theme.radiusSmall; color: "#0D0807"; border.color: Theme.lineSoft }
                }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal
            visible: page.mode === 2
            Rectangle {
                SplitView.preferredWidth: 300
                SplitView.minimumWidth: 230
                radius: Theme.radius
                color: Theme.backgroundRaised
                border.color: Theme.lineSoft
                ListView {
                    id: fileList
                    anchors.fill: parent; anchors.margins: 9
                    spacing: 6; clip: true
                    model: backend.logFiles
                    delegate: Rectangle {
                        id: fileRow
                        required property var modelData
                        width: fileList.width; height: 62
                        radius: Theme.radiusSmall; color: fileHover.hovered ? Theme.panelHover : Theme.panel; border.color: Theme.lineSoft
                        HoverHandler { id: fileHover }
                        TapHandler { onTapped: backend.loadLogFile(fileRow.modelData.name, 500) }
                        ColumnLayout {
                            anchors.fill: parent; anchors.margins: 11; spacing: 2
                            Label { Layout.fillWidth: true; text: fileRow.modelData.name; color: Theme.text; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 12; elide: Text.ElideMiddle }
                            Label { text: Math.round(Number(fileRow.modelData.size_bytes || 0) / 1024) + " KB  ·  " + (fileRow.modelData.mtime || ""); color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 9 }
                        }
                    }
                    Label { anchors.centerIn: parent; visible: fileList.count === 0; text: "アプリログはありません"; color: Theme.textMuted; font.family: Theme.uiFont }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
            Rectangle {
                SplitView.fillWidth: true
                radius: Theme.radius; color: Theme.backgroundRaised; border.color: Theme.lineSoft
                TextArea {
                    anchors.fill: parent; anchors.margins: 12
                    text: backend.logs; readOnly: true; selectByMouse: true; wrapMode: TextEdit.NoWrap
                    color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 11
                    background: Rectangle { radius: Theme.radiusSmall; color: "#0D0807"; border.color: Theme.lineSoft }
                }
            }
        }
    }

    Dialog {
        id: deleteDialog
        property string fileName: ""
        anchors.centerIn: QQC2.Overlay.overlay
        width: Math.min(420, page.width - 40)
        modal: true
        standardButtons: Dialog.NoButton
        background: Rectangle { radius: Theme.radiusLarge; color: Theme.backgroundRaised; border.color: Theme.line }
        contentItem: ColumnLayout {
            spacing: 12
            Label { text: "この会話記録を削除しますか？"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 18 }
            Label { Layout.fillWidth: true; text: "削除した記録は元に戻せません。"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11; wrapMode: Text.Wrap }
            RowLayout {
                Layout.fillWidth: true
                Item { Layout.fillWidth: true }
                BuddyButton { text: "キャンセル"; onClicked: deleteDialog.close() }
                BuddyButton { text: "削除する"; danger: true; onClicked: { page.backend.deleteHistory(deleteDialog.fileName); deleteDialog.close() } }
            }
        }
    }

    Component.onCompleted: backend.loadHistories()
}
