pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Item {
    id: page
    property var backend
    function focusAdd() { Qt.callLater(taskInput.forceActiveFocus) }
    function dueLabel(value) {
        if (!value) return "期限なし"
        const date = new Date(value)
        if (isNaN(date.getTime())) return value
        return (date.getMonth() + 1) + "/" + date.getDate() + " "
            + String(date.getHours()).padStart(2, "0") + ":" + String(date.getMinutes()).padStart(2, "0")
    }
    function overdue(task) { return task.due_at && new Date(task.due_at).getTime() < Date.now() }
    function editDueValue(value) {
        if (!value) return ""
        const date = new Date(value)
        if (isNaN(date.getTime())) return ""
        return (date.getMonth() + 1) + "/" + date.getDate() + " "
            + String(date.getHours()).padStart(2, "0") + ":" + String(date.getMinutes()).padStart(2, "0")
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Repeater {
                model: [
                    { label: "未完了", value: backend.tasks.length, tone: Theme.accent },
                    { label: "期限超過", value: backend.tasks.filter(t => page.overdue(t)).length, tone: Theme.warning },
                    { label: "だいじ", value: backend.tasks.filter(t => t.priority === "high").length, tone: Theme.magenta }
                ]
                delegate: Rectangle {
                    required property var modelData
                    Layout.fillWidth: true
                    implicitHeight: 82
                    radius: Theme.radius
                    color: Theme.panel
                    border.color: Theme.lineSoft
                    ColumnLayout {
                        anchors.fill: parent; anchors.margins: 15; spacing: 2
                        Label { text: modelData.label; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11 }
                        Label { text: modelData.value; color: modelData.tone; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 25 }
                    }
                }
            }
            Rectangle {
                Layout.fillWidth: true
                implicitHeight: 82
                radius: Theme.radius
                color: Theme.accent
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 15; spacing: 2
                    Label { text: "次の一歩"; color: "#7C4139"; font.family: Theme.uiFont; font.pixelSize: 11 }
                    Label {
                        Layout.fillWidth: true
                        text: backend.tasks.length ? (backend.tasks[0].action_hint || backend.tasks[0].title) : "いまは自由"
                        color: Theme.background; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 14
                        elide: Text.ElideRight
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: addColumn.implicitHeight + 24
            radius: Theme.radius
            color: Theme.panel
            border.color: taskInput.activeFocus ? Theme.accent : Theme.lineSoft
            ColumnLayout {
                id: addColumn
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    TextField {
                        id: taskInput
                        Layout.fillWidth: true
                        placeholderText: "例：金曜18時までに資料を確認"
                        color: Theme.text; placeholderTextColor: Theme.textMuted
                        font.family: Theme.uiFont; font.pixelSize: 14
                        selectByMouse: true
                        background: Item {}
                        onAccepted: addButton.clicked()
                    }
                    BuddyComboBox {
                        id: priority
                        model: ["ふつう", "だいじ", "あとで"]
                        property var values: ["normal", "high", "low"]
                        implicitWidth: 108
                        font.family: Theme.uiFont
                    }
                    Button {
                        id: addButton
                        text: "追加"
                        implicitWidth: 82; implicitHeight: 40
                        enabled: taskInput.text.trim().length > 0
                        onClicked: {
                            backend.addTask(taskInput.text, priority.values[priority.currentIndex], "")
                            taskInput.clear()
                        }
                        background: Rectangle { radius: Theme.radiusSmall; color: addButton.enabled ? (addButton.hovered ? Theme.accentStrong : Theme.accent) : Theme.lineSoft }
                        contentItem: Label { text: addButton.text; color: Theme.background; font.family: Theme.uiFont; font.bold: true; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                    }
                }
                Label { text: "日時・優先度を自然な日本語で書けます。登録時に最初の一歩まで自動で作ります。"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11 }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: Theme.radius
            color: Theme.backgroundRaised
            border.color: Theme.lineSoft
            ListView {
                id: taskList
                anchors.fill: parent
                anchors.margins: 10
                spacing: 8
                clip: true
                model: backend.tasks
                delegate: Rectangle {
                    id: row
                    required property var modelData
                    width: taskList.width
                    height: taskContent.implicitHeight + 24
                    radius: Theme.radiusSmall
                    color: hover.hovered ? Theme.panelHover : Theme.panel
                    border.color: page.overdue(modelData) ? Theme.warning : Theme.lineSoft
                    HoverHandler { id: hover }
                    RowLayout {
                        id: taskContent
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 12
                        Button {
                            id: doneButton
                            implicitWidth: 34; implicitHeight: 34
                            text: "✓"
                            onClicked: backend.completeTask(row.modelData.id)
                            background: Rectangle { radius: 17; color: doneButton.hovered ? Theme.green : "transparent"; border.color: Theme.green; border.width: 1 }
                            contentItem: Label { text: doneButton.text; color: doneButton.hovered ? Theme.background : Theme.green; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter; font.bold: true }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 4
                            RowLayout {
                                Layout.fillWidth: true
                                Label { Layout.fillWidth: true; text: row.modelData.title; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 15; elide: Text.ElideRight }
                                Label { text: page.dueLabel(row.modelData.due_at); color: page.overdue(row.modelData) ? Theme.warning : Theme.textMuted; font.family: Theme.monoFont; font.pixelSize: 11 }
                            }
                            Label {
                                Layout.fillWidth: true
                                text: "最初の一歩  ·  " + (row.modelData.action_hint || "小さく始める")
                                color: Theme.accent; font.family: Theme.uiFont; font.pixelSize: 12; elide: Text.ElideRight
                            }
                            Label { Layout.fillWidth: true; visible: !!row.modelData.note; text: row.modelData.note || ""; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 11; elide: Text.ElideRight }
                        }
                        Button {
                            id: stepsButton
                            text: "分解"
                            onClicked: backend.regenerateTask(row.modelData.id)
                            background: Rectangle { radius: Theme.radiusSmall; color: stepsButton.hovered ? Theme.panelHover : "transparent"; border.color: Theme.line }
                            contentItem: Label { text: stepsButton.text; color: Theme.textMuted; font.family: Theme.uiFont; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                        Button {
                            id: editButton
                            text: "編集"
                            onClicked: editDialog.openFor(row.modelData)
                            background: Rectangle { radius: Theme.radiusSmall; color: editButton.hovered ? Theme.panelHover : "transparent"; border.color: Theme.line }
                            contentItem: Label { text: editButton.text; color: Theme.text; font.family: Theme.uiFont; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                    }
                }
                Label { anchors.centerIn: parent; visible: taskList.count === 0; text: "未完了のタスクはありません"; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 15 }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }

    Dialog {
        id: editDialog
        property int taskId: 0
        anchors.centerIn: Overlay.overlay
        width: Math.min(540, page.width - 40)
        modal: true
        title: "タスクを編集"
        standardButtons: Dialog.NoButton
        function openFor(task) {
            taskId = task.id
            editTitle.text = task.title || ""
            editDue.text = page.editDueValue(task.due_at)
            editPriority.currentIndex = task.priority === "high" ? 1 : task.priority === "low" ? 2 : 0
            editNote.text = task.note || ""
            editFirst.text = task.action_hint || ""
            open()
        }
        background: Rectangle { radius: Theme.radiusLarge; color: Theme.backgroundRaised; border.color: Theme.line }
        contentItem: ColumnLayout {
            spacing: 10
            Label { text: "タスクを編集"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 21 }
            TextField { id: editTitle; Layout.fillWidth: true; placeholderText: "タイトル"; color: Theme.text; font.family: Theme.uiFont }
            TextField { id: editDue; Layout.fillWidth: true; placeholderText: "期限（空欄で削除・例：明日18時）"; color: Theme.text; font.family: Theme.uiFont }
            BuddyComboBox { id: editPriority; Layout.fillWidth: true; model: ["ふつう", "だいじ", "あとで"]; property var values: ["normal", "high", "low"] }
            TextArea { id: editNote; Layout.fillWidth: true; placeholderText: "メモ"; color: Theme.text; font.family: Theme.uiFont; wrapMode: TextEdit.Wrap }
            TextField { id: editFirst; Layout.fillWidth: true; placeholderText: "最初の一歩"; color: Theme.text; font.family: Theme.uiFont }
            RowLayout {
                Layout.fillWidth: true
                BuddyButton { text: "削除"; danger: true; onClicked: { backend.dropTask(editDialog.taskId); editDialog.close() } }
                Item { Layout.fillWidth: true }
                BuddyButton { text: "キャンセル"; onClicked: editDialog.close() }
                BuddyButton {
                    text: "保存"
                    accent: true
                    enabled: editTitle.text.trim().length > 0
                    onClicked: {
                        backend.updateTask(editDialog.taskId, editTitle.text, editDue.text, editPriority.values[editPriority.currentIndex], editNote.text, editFirst.text)
                        editDialog.close()
                    }
                }
            }
        }
    }
}
