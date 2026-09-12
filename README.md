# SCCS UAV Tracking

This repository contains the software for the SCCS Vehicle Tracking UAV project.

The project is split into two major software components:

- `ground_control/` — Ground-station software intended to run on a development PC.
- `drone_control/` — Drone-side software intended to run on the Raspberry Pi connected to the UAV hardware.

The development Docker environment is designed to give team members a consistent Python/Linux environment without requiring everyone to manually install the same dependencies.

> **Important:** The development Docker image does **not** include the Hailo hardware runtime, Hailo accelerator access, Raspberry Pi camera stack, or other drone-specific hardware dependencies. Those are handled separately on the Raspberry Pi.

---

## Repository Structure

```text
SCCS-Drone/
├── drone_control/
│   ├── main.py
│   ├── app_state.py
│   ├── centroid_tracker.py
│   ├── drone_command_ctrl.py
│   ├── hailo_inference.py
│   ├── jitter_reducer.py
│   ├── vid_cmd_data.py
│   └── config.toml
│
├── ground_control/
│   ├── main.py
│   ├── app_state.py
│   ├── data_cmd.py
│   ├── ui.py
│   ├── video.py
│   └── config.toml
│
├── Dockerfile.dev
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Development Architecture

The project currently uses two different execution environments:

```text
Development PC
Windows + WSL2 + Docker
        |
        +-- ground_control/
        +-- Python development
        +-- MAVLink development/testing
        +-- future simulation/testing

Raspberry Pi / Drone
Linux / ARM64
        |
        +-- drone_control/
        +-- Camera
        +-- Hailo accelerator
        +-- HailoRT / Hailo GStreamer plugins
        +-- Flight controller
```

The development Docker image is intended for portable development and testing on team members' computers.

The Raspberry Pi environment is hardware-specific and should be configured separately.

---

# Quick Start for Development

## 1. Prerequisites

Install the following before continuing:

- Git
- Docker Desktop
- WSL2 with Ubuntu if using Windows

Docker Desktop should be running before using Docker from WSL.

### Verify Docker

From WSL:

```bash
docker --version
```

Then:

```bash
docker run --rm hello-world
```

If Docker is configured correctly, you should see:

```text
Hello from Docker!
```

---

## 2. Clone the Repository

If you have not already cloned the project:

```bash
git clone https://github.com/Frodo1526/SCCS-UAV-Tracking.git
```

Then:

```bash
cd SCCS-UAV-Tracking
```

If you already have the repository:

```bash
git pull
```

---

## 3. Build the Development Docker Image

From the root of the repository:

```bash
docker build -f Dockerfile.dev -t sccs-drone-dev .
```

This builds the development image using:

- Ubuntu 24.04
- Python
- NumPy
- SciPy
- Pygame
- msgpack
- MAVSDK
- pymavlink
- GObject Introspection
- GStreamer
- other dependencies listed in `requirements.txt`

---

## 4. Run the Development Container

Recommended command:

```bash
docker run --rm -it -v "$(pwd):/app" sccs-drone-dev
```

This opens an interactive shell inside the container.

The `-v "$(pwd):/app"` option bind-mounts your local repository into the container.

This means:

```text
Files edited on your PC
        |
        v
Immediately visible inside /app
```

You do not need to rebuild the Docker image every time you change a Python file.

Once inside the container, the prompt should look similar to:

```text
root@container-id:/app#
```

---

## 5. Verify the Development Environment

Inside the container:

```bash
python --version
```

Then verify the primary portable dependencies:

```bash
python -c "import numpy, scipy, pygame, msgpack, mavsdk, pymavlink, gi; print('Environment OK')"
```

If everything is configured correctly, you should see:

```text
Environment OK
```

---

# Ground Control

The ground-control application is located at:

```text
ground_control/main.py
```

It is responsible for functionality such as:

- displaying the video stream
- rendering the UI
- receiving telemetry
- displaying tracked objects
- allowing target selection
- sending commands back to the drone

The ground-control application may require additional display/network configuration when running from inside Docker.

---

# Drone Control

The drone-side application is located at:

```text
drone_control/main.py
```

It is intended to run on the Raspberry Pi attached to the UAV.

The current drone-side pipeline expects hardware/software including:

- Raspberry Pi / ARM64 Linux
- Raspberry Pi-compatible camera
- GStreamer
- HailoRT
- Hailo GStreamer plugins
- Hailo-8 or Hailo-8L accelerator
- compatible `.hef` model files
- flight-controller connectivity

The inherited code currently contains hardware-specific paths such as:

```text
/usr/lib/aarch64-linux-gnu/hailo/
```

and uses GStreamer elements such as:

```text
hailonet
hailofilter
libcamerasrc
```

For this reason, `drone_control/main.py` is **not expected to run fully inside the normal PC development container**.

This is intentional.

---

# Why Hailo Is Not Included in the Development Container

Hailo is not simply a normal Python dependency.

The full Hailo stack includes:

```text
Python Hailo API
        |
        v
HailoRT userspace runtime
        |
        v
Hailo GStreamer plugins
        |
        v
Linux Hailo driver
        |
        v
Physical Hailo accelerator
```

The drone software also expects ARM64-native libraries and Raspberry Pi camera support.

Most team development computers are x86-64 systems and do not contain a Hailo accelerator.

For that reason:

- `Dockerfile.dev` is the portable development environment.
- The Raspberry Pi/Hailo environment is configured separately.
- A future Pi-specific Docker image may be added once the physical hardware stack is validated.

---

# Python Dependencies

Python dependencies are stored in:

```text
requirements.txt
```

Current dependencies include packages such as:

```text
numpy
scipy
pygame
msgpack
mavsdk
pymavlink
```

and their transitive dependencies.

If a new Python package is required:

```bash
source .venv/bin/activate
python -m pip install <package-name>
python -m pip freeze > requirements.txt
```

Then rebuild the Docker image:

```bash
docker build -f Dockerfile.dev -t sccs-drone-dev .
```

Commit the updated `requirements.txt` so everyone receives the same environment.

---

# Git Workflow

Before beginning work:

```bash
git pull
```

After making changes:

```bash
git status
```

Stage the files you changed:

```bash
git add <files>
```

Commit:

```bash
git commit -m "Describe the change"
```

Push:

```bash
git push
```

Team members should avoid committing:

```text
.venv/
__pycache__/
.env
IDE-specific files
logs
```

These are already covered by `.gitignore`.

---

# Useful Docker Commands

## Show Docker images

```bash
docker images
```

## Show running containers

```bash
docker ps
```

## Show all containers

```bash
docker ps -a
```

## Remove an old image

```bash
docker rmi <image-name>
```

Example:

```bash
docker rmi sccs-drone-dev
```

## Rebuild the development image

```bash
docker build -f Dockerfile.dev -t sccs-drone-dev .
```

## Start the development environment

```bash
docker run --rm -it -v "$(pwd):/app" sccs-drone-dev
```

## Exit the container

```bash
exit
```

---

# Troubleshooting

## Docker command not found in WSL

Make sure Docker Desktop is running and WSL integration is enabled.

In Docker Desktop:

```text
Settings
-> Resources
-> WSL Integration
```

Enable the Ubuntu distribution you are using.

---

## Docker cannot find `sccs-drone-dev`

Check:

```bash
docker images
```

If it is missing, rebuild:

```bash
docker build -f Dockerfile.dev -t sccs-drone-dev .
```

---

## Python import is missing

If the missing package is a normal Python package:

```bash
python -m pip install <package-name>
```

Then update:

```bash
python -m pip freeze > requirements.txt
```

Do not blindly install packages for imports such as:

```python
import gi
import hailo
```

These may depend on operating-system packages or hardware-specific SDKs.

---

## `import hailo` fails

This is expected on a normal development PC.

The current drone software expects the Hailo runtime and physical accelerator.

Do not install an unrelated package named `hailo` from PyPI simply to remove the import error.

---

## `drone_control/main.py` fails on a development PC

This is also expected if the error involves:

- `hailo`
- `hailonet`
- `hailofilter`
- `libcamerasrc`
- `.hef` model files
- ARM64 Hailo libraries
- camera hardware

The full drone-side application is intended to run on the Raspberry Pi hardware environment.

---

# Future Development

Planned improvements may include:

- Raspberry Pi-specific deployment instructions
- Pi-specific Docker image
- PX4 or ArduPilot SITL integration
- simulated camera/inference input
- mock or CPU-based detection backend
- automated dependency checks
- automated Docker builds
- separation of hardware-specific inference from tracking/control logic

A future development goal is to allow the same tracking and control logic to operate with either:

```text
Hailo hardware inference
        or
simulated / CPU inference
```

without changing the rest of the application.

---

# Project Notes

The current project is inherited from previous development teams. Some modules are hardware-specific and may require cleanup or refactoring as the current team becomes familiar with the architecture.

When modifying existing code, prioritize maintaining compatibility with the working Raspberry Pi/Hailo implementation while improving portability and testability.
