# SO-100 Robotic Arm: Manjaro Linux Teleoperation & Troubleshooting Guide

This guide details the complete troubleshooting steps and configuration process required to set up an **SO-100 robotic arm (6 Feetech servos)** for keyboard-based teleoperation using **LeRobot (v0.6.2)** on **Manjaro Linux**.

---

## 1. Hardware & System Setup Overview

* **Hardware:** SO-100 robotic arm with 6 Feetech serial bus servos, connected via USB to `/dev/ttyACM0`.
* **OS / Environment:** Manjaro GNOME (Wayland by default), running within a Python virtual environment.
* **Core Software:** `lerobot`, `pynput` (for keyboard listening), Feetech motor communication drivers.

---

## 2. Resolving Display Server Restrictions (Wayland vs. X11)

By default, Manjaro GNOME runs on **Wayland**, which security-blocks global keyboard interception libraries like `pynput`. The teleoperation scripts will fail to hook keypresses unless an X11 environment or proper display forwarding is established.

### Method A: Forcing X11 Session via GDM (If Auto-Login permits)

1. Edit the GDM custom configuration file:

   ```bash
   sudo nano /etc/gdm/custom.conf
   ```

2. Ensure Wayland is explicitly disabled and auto-login is adjusted if necessary:

   Plaintext

   ```text
   [daemon]
   WaylandEnable=false
   AutomaticLoginEnable=False
   ```

3. Reboot the system and select **GNOME on Xorg** at the login screen.
4. Verify the session type:

   Bash

   ```bash
   echo $XDG_SESSION_TYPE
   ```

   *Expected output:* `x11`

### Method B: Root-Level Display Forwarding (Ultimate Override)

If the system forcibly falls back to Wayland due to stubborn auto-login configurations, run the script with root privileges while explicitly forwarding the active display socket and Xauthority:

Bash

```bash
sudo DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY ./venv/bin/lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=keyboard
```

## 3. Troubleshooting Motor Bus & `StopIteration` Errors

### Issue 1: Serial Bus Status Packets / `StopIteration` during `sync_write`

When the script initializes, it sends a `TorqueEnable` command to all motors simultaneously.

- **Symptom:** `StopIteration` traceback inside `motors_bus.py` at line 1248 (`model = next(iter(models))`), or errors like `failed to write torqueenable on id = X`.
- **Root Cause:**
  1. **Voltage Drop / Brownout:** Drawing power for all 6 servos simultaneously over a weak USB or underpowered 12V line causes the bus controller to reset, returning an empty list of connected motors.
  2. **Loose Daisy-Chain Wiring:** A loose 3-pin connector on one of the servos (frequently observed shifting between ID 3, 4, or 6) corrupts data packets across the serial chain.

### Solution for Bus Stability

1. **Physical Rest Position:** Before executing the teleop script, manually fold and rest the arm entirely flat on the desk so motors do not fight gravity upon torque engagement.
2. **Dedicated Power:** Ensure a robust external **12V power supply (at least 3A–5A)** is plugged in and switched on *before* connecting the USB cable to the computer.
3. **Check Connections:** Inspect and firmly reseat the 3-pin data/power cables connecting the affected servos.

## 4. Successful Execution Flow

Once the display server forwarding is active and the motor bus is stable, launch the teleoperation command:

Bash

```bash
cd /home/yash/Documents/ARC/so100/lerobot
source venv/bin/activate
sudo DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY ./venv/bin/lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --teleop.type=keyboard
```

### Key Bindings

Once the terminal logs confirm `pynput is available - enabling local keyboard listener`, keep the terminal window in focus and use the mapped keys to drive the individual joints of the SO-100 arm.
