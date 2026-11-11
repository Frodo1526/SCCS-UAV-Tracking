# Pi directory layout

Files and directories on the air Pi that provisioning, the service, and logging touch.

```
/
├── home/capstone-pi/
│   └── SCCS-UAV-Tracking/                   
│       ├── common/sccs_common/             
│       ├── drone_air/                     
│       │   ├── air_boot.py                # systemd entry point
│       │   ├── main.py                    # tracking algorithms, forwarding code, log/error checking, etc
│       │   └── config.toml
│       └── hardware_backups/pi_prov/
│           ├── setup.sh                   # initial provisioning and OS status/preflight verification  
│           ├── sccs_air.service           # systemd service
│           └── README.md
├── etc/systemd/system/
│   └── sccs_air.service                   # this is where the live service reads from... 
├── boot/firmware/
│   ├── config.txt                         # camera setup  
│   └── config.txt.bak.<YYYYmmdd-HHMMSS>   # backup created by setup.sh
└── var/log/flight_logs/                 
    ├── <air_boot log>                  
    └── setup_verify.log                   # standard logfile for all in-air operations/checks 
```
