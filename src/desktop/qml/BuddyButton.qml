import QtQuick
import QtQuick.Controls
import "."

Button {
    id: control
    property bool accent: false
    property bool danger: false
    implicitHeight: 38
    implicitWidth: Math.max(76, label.implicitWidth + 28)
    background: Rectangle {
        radius: Theme.radiusSmall
        color: !control.enabled ? Theme.lineSoft
            : control.danger ? (control.hovered ? Theme.warning : "transparent")
            : control.accent ? (control.hovered ? Theme.accentStrong : Theme.accent)
            : control.hovered ? Theme.panelHover : Theme.panel
        border.color: control.accent ? Theme.accent : control.danger ? Theme.warning : Theme.line
        border.width: 1
        Behavior on color { ColorAnimation { duration: Theme.motionFast } }
    }
    contentItem: Label {
        id: label
        text: control.text
        color: control.accent || (control.danger && control.hovered) ? Theme.background : control.danger ? Theme.warning : Theme.text
        font.family: Theme.uiFont
        font.bold: true
        font.pixelSize: 12
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}
