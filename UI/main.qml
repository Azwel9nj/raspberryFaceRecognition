import QtQuick
import QtQuick.Controls.Basic
ApplicationWindow {
    visible: true
    width: 1000
    height: 600
    x: screen.desktopAvailableWidth - width - 12
    y: screen.desktopAvailableHeight - height - 48
    title: "HelloApp"
    
    property string currTime: "00:00:00"
    property QtObject backend
    Rectangle {
        anchors.fill: parent
        Image {
            sourceSize.width: parent.width
            sourceSize.height: parent.height
            source: "../images/sajad-fi-lYBrEobCW5I-unsplash.jpg"
            fillMode: Image.PreserveAspectFit
        }
        Text {
            anchors {
                bottom: parent.bottom
                bottomMargin: 12
                left: parent.left
                leftMargin: 12
            }
            text: currTime
            font.pixelSize: 48
            color: "white"
        }
    }

    Connections {
        target: backend
        function onUpdated(msg) {
            currTime = msg;
        }
    }
}