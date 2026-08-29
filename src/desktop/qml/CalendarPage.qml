pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Controls as QQC2
import QtQuick.Layouts
import "."

Item {
    id: page
    property var backend
    property date shownMonth: new Date()
    property string selectedDate: formatDate(new Date())

    function pad(value) { return String(value).padStart(2, "0") }
    function formatDate(value) {
        return value.getFullYear() + "-" + pad(value.getMonth() + 1) + "-" + pad(value.getDate())
    }
    function monthStart() { return formatDate(new Date(shownMonth.getFullYear(), shownMonth.getMonth(), 1)) }
    function monthEnd() { return formatDate(new Date(shownMonth.getFullYear(), shownMonth.getMonth() + 1, 0)) }
    function cellDate(index) {
        const first = new Date(shownMonth.getFullYear(), shownMonth.getMonth(), 1)
        return new Date(shownMonth.getFullYear(), shownMonth.getMonth(), index - first.getDay() + 1)
    }
    function eventsOn(day) {
        return (backend.calendarEvents || []).filter(event => String(event.start || "").slice(0, 10) === day)
    }
    function activate() { backend.loadCalendar(monthStart(), monthEnd()) }
    function moveMonth(delta) {
        shownMonth = new Date(shownMonth.getFullYear(), shownMonth.getMonth() + delta, 1)
        selectedDate = monthStart()
        activate()
    }

    RowLayout {
        anchors.fill: parent
        spacing: 12

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumWidth: 400
            radius: Theme.radius
            color: Theme.backgroundRaised
            border.color: Theme.lineSoft
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 8
                RowLayout {
                    Layout.fillWidth: true
                    BuddyButton { text: "‹"; onClicked: page.moveMonth(-1) }
                    Label { Layout.fillWidth: true; text: page.shownMonth.getFullYear() + "年 " + (page.shownMonth.getMonth() + 1) + "月"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 20; horizontalAlignment: Text.AlignHCenter }
                    BuddyButton { text: "今日"; onClicked: { page.shownMonth = new Date(); page.selectedDate = page.formatDate(new Date()); page.activate() } }
                    BuddyButton { text: "›"; onClicked: page.moveMonth(1) }
                }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 7
                    columnSpacing: 4
                    Repeater {
                        model: ["日", "月", "火", "水", "木", "金", "土"]
                        delegate: Label {
                            required property string modelData
                            Layout.fillWidth: true
                            text: modelData
                            color: modelData === "日" ? Theme.magenta : Theme.textMuted
                            font.family: Theme.uiFont; font.pixelSize: 11
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                }
                GridLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    columns: 7
                    columnSpacing: 4
                    rowSpacing: 4
                    Repeater {
                        model: 42
                        delegate: Rectangle {
                            id: dayCell
                            required property int index
                            property var value: page.cellDate(index)
                            property string dayKey: page.formatDate(value)
                            property bool currentMonth: value.getMonth() === page.shownMonth.getMonth()
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: Theme.radiusSmall
                            color: page.selectedDate === dayKey ? Theme.panelStrong : dayHover.hovered ? Theme.panelHover : Theme.panel
                            border.color: page.selectedDate === dayKey ? Theme.accent : Theme.lineSoft
                            opacity: currentMonth ? 1 : 0.42
                            HoverHandler { id: dayHover }
                            TapHandler { onTapped: page.selectedDate = dayCell.dayKey }
                            Label { anchors.left: parent.left; anchors.top: parent.top; anchors.margins: 7; text: dayCell.value.getDate(); color: Theme.text; font.family: Theme.monoFont; font.pixelSize: 11 }
                            Row {
                                anchors.left: parent.left; anchors.bottom: parent.bottom; anchors.margins: 7; spacing: 3
                                Repeater {
                                    model: Math.min(4, page.eventsOn(dayCell.dayKey).length)
                                    Rectangle { required property int index; width: 6; height: 6; radius: 3; color: index === 0 ? Theme.accent : Theme.green }
                                }
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.preferredWidth: 280
            Layout.minimumWidth: 245
            Layout.fillHeight: true
            radius: Theme.radius
            color: Theme.backgroundRaised
            border.color: Theme.lineSoft
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 9
                RowLayout {
                    Layout.fillWidth: true
                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 1
                        Label { text: page.selectedDate; color: Theme.text; font.family: Theme.monoFont; font.bold: true; font.pixelSize: 14 }
                        Label { text: backend.calendarWritable ? "Google Calendarへ反映" : "予定は読み取り専用"; color: backend.calendarWritable ? Theme.green : Theme.warning; font.family: Theme.uiFont; font.pixelSize: 10 }
                    }
                    BuddyButton { text: "＋ 予定"; accent: true; enabled: backend.calendarWritable; onClicked: eventDialog.openNew(page.selectedDate) }
                }
                ListView {
                    id: eventList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 7
                    clip: true
                    model: page.eventsOn(page.selectedDate)
                    delegate: Rectangle {
                        id: eventRow
                        required property var modelData
                        width: eventList.width
                        height: eventColumn.implicitHeight + 22
                        radius: Theme.radiusSmall
                        color: eventHover.hovered ? Theme.panelHover : Theme.panel
                        border.color: Theme.lineSoft
                        HoverHandler { id: eventHover }
                        TapHandler { onTapped: eventDialog.openEdit(eventRow.modelData) }
                        ColumnLayout {
                            id: eventColumn
                            anchors.fill: parent; anchors.margins: 11; spacing: 3
                            Label { Layout.fillWidth: true; text: eventRow.modelData.title || "予定"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 13; wrapMode: Text.Wrap }
                            Label { text: String(eventRow.modelData.start || "").indexOf("T") >= 0 ? String(eventRow.modelData.start).slice(11, 16) : "終日"; color: Theme.accent; font.family: Theme.monoFont; font.pixelSize: 10 }
                            Label { visible: !!eventRow.modelData.location; text: eventRow.modelData.location || ""; color: Theme.textMuted; font.family: Theme.uiFont; font.pixelSize: 10 }
                        }
                    }
                    Label { anchors.centerIn: parent; visible: eventList.count === 0; text: "この日の予定はありません"; color: Theme.textMuted; font.family: Theme.uiFont }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
        }
    }

    Dialog {
        id: eventDialog
        property string eventId: ""
        anchors.centerIn: QQC2.Overlay.overlay
        width: Math.min(500, page.width - 40)
        modal: true
        standardButtons: Dialog.NoButton
        function openNew(day) {
            eventId = ""; eventTitle.text = ""; eventDate.text = day; eventTime.text = ""; eventDuration.value = 60; eventLocation.text = ""; eventDescription.text = ""; open()
        }
        function openEdit(event) {
            eventId = String(event.event_id || ""); eventTitle.text = event.title || ""; eventDate.text = String(event.start || "").slice(0, 10); eventTime.text = String(event.start || "").indexOf("T") >= 0 ? String(event.start).slice(11, 16) : ""; eventLocation.text = event.location || ""; eventDescription.text = event.description || ""; open()
        }
        background: Rectangle { radius: Theme.radiusLarge; color: Theme.backgroundRaised; border.color: Theme.line }
        contentItem: ColumnLayout {
            spacing: 9
            Label { text: eventDialog.eventId ? "予定を編集" : "予定を追加"; color: Theme.text; font.family: Theme.uiFont; font.bold: true; font.pixelSize: 20 }
            TextField { id: eventTitle; Layout.fillWidth: true; placeholderText: "予定名"; color: Theme.text; font.family: Theme.uiFont }
            RowLayout {
                Layout.fillWidth: true
                TextField { id: eventDate; Layout.fillWidth: true; placeholderText: "YYYY-MM-DD"; color: Theme.text; font.family: Theme.monoFont }
                TextField { id: eventTime; Layout.preferredWidth: 110; placeholderText: "HH:MM（任意）"; color: Theme.text; font.family: Theme.monoFont }
                SpinBox { id: eventDuration; from: 5; to: 1440; value: 60; stepSize: 5; editable: true }
            }
            TextField { id: eventLocation; Layout.fillWidth: true; placeholderText: "場所（任意）"; color: Theme.text; font.family: Theme.uiFont }
            TextArea { id: eventDescription; Layout.fillWidth: true; placeholderText: "メモ（任意）"; color: Theme.text; font.family: Theme.uiFont; wrapMode: TextEdit.Wrap }
            RowLayout {
                Layout.fillWidth: true
                BuddyButton { visible: !!eventDialog.eventId; text: "削除"; danger: true; onClicked: { backend.deleteCalendarEvent(eventDialog.eventId); eventDialog.close() } }
                Item { Layout.fillWidth: true }
                BuddyButton { text: "キャンセル"; onClicked: eventDialog.close() }
                BuddyButton {
                    text: "保存"; accent: true
                    enabled: eventTitle.text.trim().length > 0 && eventDate.text.trim().length > 0
                    onClicked: {
                        if (eventDialog.eventId) backend.updateCalendarEvent(eventDialog.eventId, { title: eventTitle.text, date: eventDate.text, time: eventTime.text, duration_min: eventDuration.value, location: eventLocation.text, description: eventDescription.text })
                        else backend.createCalendarEvent(eventTitle.text, eventDate.text, eventTime.text, eventDuration.value, eventLocation.text, eventDescription.text)
                        eventDialog.close()
                    }
                }
            }
        }
    }
}
