pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Item {
    id: page
    property var backend

    Flickable {
        anchors.fill: parent
        contentWidth: width
        contentHeight: content.implicitHeight
        clip: true
        ScrollBar.vertical: ScrollBar {}

        ColumnLayout {
            id: content
            width: parent.width
            spacing: 12
            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: 118
                    radius: Theme.radius
                    color: Theme.accent
                    RowLayout {
                        anchors.fill: parent; anchors.margins: 20; spacing: 18
                        Rectangle { implicitWidth: 66; implicitHeight: 66; radius: 33; color: Theme.background; Label { anchors.centerIn: parent; text: "✦"; color: Theme.accent; font.pixelSize: 27 } }
                        ColumnLayout {
                            Layout.fillWidth: true; spacing: 3
                            Label { text: "現在の相棒ランク"; color: "#7C4139"; font.family: Theme.uiFont; font.pixelSize: 11 }
                            Label { text: (backend.game.rank && backend.game.rank.name) || "相棒"; color: Theme.background; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 27 }
                            Label { text: backend.game.rank && backend.game.rank.next ? "次は Lv." + backend.game.rank.next.level + "「" + backend.game.rank.next.name + "」" : "最高ランク"; color: "#7C4139"; font.family: Theme.uiFont; font.pixelSize: 11 }
                        }
                    }
                }
                Repeater {
                    model: [
                        { label: "育ちポイント", value: Number(backend.game.points || 0).toLocaleString() + " GP" },
                        { label: "解除した実績", value: (backend.game.unlocked_badges || 0) + " / " + ((backend.game.badges || []).length) }
                    ]
                    delegate: Rectangle {
                        required property var modelData
                        Layout.preferredWidth: 210
                        implicitHeight: 118
                        radius: Theme.radius
                        color: Theme.panel
                        border.color: Theme.lineSoft
                        ColumnLayout {
                            anchors.centerIn: parent; spacing: 5
                            Label { Layout.alignment: Qt.AlignHCenter; text: modelData.label; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11 }
                            Label { Layout.alignment: Qt.AlignHCenter; text: modelData.value; color: Theme.text; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 22 }
                        }
                    }
                }
            }

            Label { text: "今日のクエスト"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 18 }
            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                Repeater {
                    model: backend.game.missions || []
                    delegate: Rectangle {
                        id: missionCard
                        required property var modelData
                        Layout.fillWidth: true
                        implicitHeight: 100
                        radius: Theme.radius
                        color: modelData.complete ? Theme.panelStrong : Theme.panel
                        border.color: modelData.complete ? Theme.green : Theme.lineSoft
                        ColumnLayout {
                            anchors.fill: parent; anchors.margins: 13; spacing: 4
                            RowLayout {
                                Layout.fillWidth: true
                                Label { Layout.fillWidth: true; text: missionCard.modelData.name; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13 }
                                BuddyButton {
                                    visible: missionCard.modelData.complete && !missionCard.modelData.claimed
                                    text: "+" + missionCard.modelData.reward
                                    accent: true
                                    onClicked: backend.claimMission(missionCard.modelData.id)
                                }
                                Label { visible: missionCard.modelData.claimed; text: "受取済"; color: Theme.green; font.family: Theme.uiFont; font.pixelSize: 10 }
                            }
                            Label { text: missionCard.modelData.detail; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                            ProgressBar {
                                Layout.fillWidth: true
                                from: 0; to: Math.max(1, missionCard.modelData.target); value: missionCard.modelData.current
                                background: Rectangle { implicitHeight: 5; radius: 3; color: Theme.lineSoft }
                                contentItem: Item { implicitHeight: 5; Rectangle { width: parent.width * Math.min(1, missionCard.modelData.current / Math.max(1, missionCard.modelData.target)); height: parent.height; radius: 3; color: Theme.green } }
                            }
                        }
                    }
                }
            }

            Label { text: "コレクション"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 18 }
            GridLayout {
                Layout.fillWidth: true
                columns: width > 900 ? 3 : 2
                columnSpacing: 10
                rowSpacing: 10
                Repeater {
                    model: backend.game.badges || []
                    delegate: Rectangle {
                        id: badgeCard
                        required property var modelData
                        Layout.fillWidth: true
                        implicitHeight: 150
                        radius: Theme.radius
                        color: modelData.unlocked ? Theme.panelStrong : Theme.panel
                        border.color: modelData.unlocked ? Theme.accent : Theme.lineSoft
                        opacity: modelData.unlocked ? 1 : 0.72
                        ColumnLayout {
                            anchors.fill: parent; anchors.margins: 15; spacing: 5
                            RowLayout {
                                Layout.fillWidth: true
                                Rectangle { implicitWidth: 36; implicitHeight: 36; radius: 18; color: badgeCard.modelData.unlocked ? Theme.magenta : Theme.lineSoft; Label { anchors.centerIn: parent; text: badgeCard.modelData.unlocked ? badgeCard.modelData.mark : "?"; color: Theme.text; font.pixelSize: 17 } }
                                Item { Layout.fillWidth: true }
                                Label { text: badgeCard.modelData.unlocked ? "解除済み" : "挑戦中"; color: badgeCard.modelData.unlocked ? Theme.accent : Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                            }
                            Label { text: badgeCard.modelData.name; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 15 }
                            Label { text: badgeCard.modelData.detail; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11 }
                            Item { Layout.fillHeight: true }
                            ProgressBar {
                                Layout.fillWidth: true
                                from: 0; to: Math.max(1, badgeCard.modelData.target); value: badgeCard.modelData.current
                                background: Rectangle { implicitHeight: 6; radius: 3; color: Theme.lineSoft }
                                contentItem: Item { implicitHeight: 6; Rectangle { width: parent.width * Math.min(1, badgeCard.modelData.current / Math.max(1, badgeCard.modelData.target)); height: parent.height; radius: 3; color: Theme.accent } }
                            }
                            Label { Layout.alignment: Qt.AlignRight; text: Math.min(badgeCard.modelData.current, badgeCard.modelData.target) + " / " + badgeCard.modelData.target + " " + badgeCard.modelData.unit; color: Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 10 }
                        }
                    }
                }
            }
            Item { Layout.fillWidth: true; implicitHeight: 20 }
        }
    }

    Component.onCompleted: backend.loadGame()
}
