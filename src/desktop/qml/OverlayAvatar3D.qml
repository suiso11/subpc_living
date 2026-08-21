pragma ComponentBehavior: Bound
import QtQuick
import QtQuick3D

Item {
    id: avatar3D

    property url modelUrl
    property bool shrink: false
    property bool dimmed: false
    property bool spinning: true

    signal loadFailed()

    opacity: dimmed ? 0.35 : 1.0

    View3D {
        id: view3D
        anchors.fill: parent

        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0, 100, 350)
            eulerRotation: Qt.vector3d(-10, 0, 0)
        }

        DirectionalLight {
            position: Qt.vector3d(0, 200, 300)
            eulerRotation: Qt.vector3d(-45, 0, 0)
            brightness: 1.0
        }

        SpotLight {
            position: Qt.vector3d(0, 300, 200)
            eulerRotation: Qt.vector3d(-60, 0, 0)
            brightness: 0.5
            coneAngle: 60
            innerConeAngle: 45
        }

        Node {
            id: modelRoot
            scale: Qt.vector3d(1, 1, 1)

            SequentialAnimation on scale.y {
                running: !avatar3D.shrink && !avatar3D.dimmed
                loops: Animation.Infinite
                NumberAnimation { from: 1.0; to: 1.03; duration: 1500; easing.type: Easing.InOutSine }
                NumberAnimation { from: 1.03; to: 1.0; duration: 1500; easing.type: Easing.InOutSine }
            }

            RotationAnimation {
                target: modelRoot
                property: "eulerRotation.y"
                from: 0
                to: 360
                duration: 12000
                loops: Animation.Infinite
                running: avatar3D.spinning && !avatar3D.shrink
                direction: RotationAnimation.Shortest
            }

            RuntimeLoader {
                id: loader
                source: avatar3D.modelUrl
                onStatusChanged: {
                    if (status === RuntimeLoader.Failed) {
                        avatar3D.loadFailed()
                    }
                }
            }
        }
    }
}
