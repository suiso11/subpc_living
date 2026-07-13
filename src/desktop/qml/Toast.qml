import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "."

Rectangle {
    id: root
    property string title: ""
    property string detail: ""
    property bool showing: false
    width: 360
    implicitHeight: content.implicitHeight + 30
    radius: Theme.radius
    color: Theme.panelStrong
    border.color: Theme.line
    border.width: 1
    opacity: showing ? 1 : 0
    y: showing ? 20 : 4
    visible: opacity > 0

    Behavior on opacity { NumberAnimation { duration: Theme.motionFast } }
    Behavior on y { NumberAnimation { duration: Theme.motion } }

    function show(messageTitle, messageDetail) {
        title = messageTitle
        detail = messageDetail
        showing = true
        timer.restart()
    }

    ColumnLayout {
        id: content
        anchors.fill: parent
        anchors.margins: 15
        spacing: 4
        Label {
            text: root.title
            color: Theme.text
            font.family: Theme.uiFont
            font.bold: true
            font.pixelSize: 14
        }
        Label {
            Layout.fillWidth: true
            text: root.detail
            color: Theme.textMuted
            font.family: Theme.uiFont
            font.pixelSize: 12
            wrapMode: Text.Wrap
            visible: text.length > 0
        }
    }
    Timer { id: timer; interval: 3600; onTriggered: root.showing = false }
}
