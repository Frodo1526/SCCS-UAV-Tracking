Tello Drone Setup and Connection Guide
1. Importing the Code
Download the Tello Control project from the repository.
Install the required external libraries:
```term
pip install djitellopy opencv-python ultralytics numpy scipy
```
---
2. Connecting to the Drone
Install the Battery
Insert a charged battery into the drone.
Note: One of the batteries is not functioning correctly.
I have been placing the known-good battery in the slot farther away.
Press the button on the side of the drone.
The LED should begin blinking. Wait a few seconds for the drone to finish starting up.
Run the Program
Run the Tello control code.
A successful connection should produce log output similar to:
```text
Sending command: command
Response: ok
```
After connecting successfully, the video feed should open.
---
Troubleshooting Connection Issues
If you see something similar to:
```text
Sending command: command
Response: "command"
```
or receive no response at all, something is preventing communication with the drone.
Check for an IP Address Conflict
The Tello drone uses:
```text
192.168.10.1
```
Make sure another program or network adapter is not already using or interfering with this address.
Open Command Prompt as Administrator and run:
```cmd
netstat -anb | find "192.168.10.1"
```
(Good luck Linux users, I don't know the command, probably just grep)

If you receive output from the command, another program or service may be interfering with communication.
Find the program using the address and change or disable its network configuration.
**Example:** On my machine, VMware was using `192.168.10.1`, which prevented communication with the Tello. Changing VMware's network address resolved the issue.