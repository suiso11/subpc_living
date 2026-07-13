pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import "."

ComboBox {
    id: control
    implicitHeight: 38
    implicitWidth: 130
    leftPadding: 12
    rightPadding: 30
    font.family: Theme.uiFont
    font.pixelSize: 12
    background: Rectangle {
        radius: Theme.radiusSmall
        color: Theme.panel
        border.color: control.activeFocus ? Theme.accent : Theme.line
    }
    contentItem: Text {
        leftPadding: 0
        rightPadding: control.indicator.width + control.spacing
        text: control.displayText
        font: control.font
        color: Theme.text
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    indicator: Label {
        x: control.width - width - 11
        y: Math.round((control.height - height) / 2)
        text: "⌄"
        color: Theme.textMuted
        font.pixelSize: 14
    }
    delegate: ItemDelegate {
        id: option
        required property var modelData
        required property int index
        width: control.width
        contentItem: Text {
            text: option.modelData
            color: option.highlighted ? Theme.background : Theme.text
            font: control.font
            verticalAlignment: Text.AlignVCenter
        }
        highlighted: control.highlightedIndex === option.index
        background: Rectangle {
            radius: 7
            color: option.highlighted ? Theme.accent : option.hovered ? Theme.panelHover : Theme.panel
        }
    }
    popup: Popup {
        y: control.height + 5
        width: control.width
        implicitHeight: contentItem.implicitHeight + 10
        padding: 5
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: control.popup.visible ? control.delegateModel : null
            currentIndex: control.highlightedIndex
            ScrollIndicator.vertical: ScrollIndicator {}
        }
        background: Rectangle { radius: Theme.radiusSmall; color: Theme.panel; border.color: Theme.line }
    }
}
