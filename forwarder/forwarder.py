from cProfile import label
import socket
import threading
import time
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DRONE_IP = "10.5.0.2" # Drone's IP address on ad-hoc wfb-ng network
LAPTOP_IP = "10.55.0.2" #Laptop's IP address on ethernet-over-usb connection
VIDEO_PORT = "5602" # Port for video stream (UDP)
DATA_PORT = "5601" # Port for telemetry data (UDP)
COMMAND_PORT = "5603" # Port for command/control (TCP)
UDP_BUFFER_SIZE = 65507 # Maximum safe UDP payload size (65,535 - 8 byte header - 20 byte IP header)s
running = True # Global flag for thread control, set to False on interrupts

def forward_udp(listen_port, dest_ip, dest_port, name):
    '''
    Runs a blocking loop that receives UDP datagrams on listen_port (bound
    to all interfaces, with an enlarged 4 MiB receive buffer) and forwards
    each one verbatim to (dest_ip, dest_port). Intended to relay video and
    telemetry traffic between the drone's ad-hoc network and the ground
    laptop. Logs a running packet count every 500 forwarded packets, and
    continues past individual socket errors (pausing 0.1s) as long as the
    module-level `running` flag is still True.

    Args:
        listen_port (int): Local UDP port to receive on.
        dest_ip (str): Destination IPv4 address to forward packets to.
        dest_port (int): Destination UDP port to forward packets to.
        name (str): Label used in log messages to identify this forwarding
            stream (e.g. "VIDEO" or "DATA").

    Returns:
        None: This function does not return a value; it runs until the
            module-level `running` flag becomes False, then closes both
            sockets.
    '''
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow for address reuse, avoiding restart errors 
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024) # 4MiB Buffer
    rx.bind(('0.0.0.0', listen_port)) #bind socket to all interfaces, externally
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Transmitting socket
    
    logging.info(f'{name} UDP: {listen_port} -> {dest_ip}:{dest_port}') 

    packets = 0 # Packet count for logging
    while running:
        try:
            data, addr = rx.recvfrom(UDP_BUFFER_SIZE) # Read UDP packet
            tx.sendto(data, (dest_ip, dest_port)) # Forward to destination
            packets += 1 # Increment packet count
            if packets % 500 == 0: # Log every 500 packets
                logging.info(f'{name} forwarded {packets} packets')
        except socket.timeout:
            logging.warning('UDP socket timeout')
            continue
        except Exception as e:
            if running:
                logging.error(f'{name} error: {e}')
                time.sleep(0.1)
    
    rx.close() 
    tx.close()
    logging.info(f'{name} stopped. Forwarded {packets} packets')
    
def handle_tcp_connection(conn, addr, drone_ip, drone_port, name):
    '''
    Handles a single incoming TCP connection from the ground station by
    opening a corresponding TCP connection to the drone and relaying data
    bidirectionally between the two, each direction on its own thread,
    until either side closes or a 10-second inactivity timeout is hit on
    either socket.

    Args:
        conn (socket.socket): The already-accepted TCP connection from the
            ground station/laptop.
        addr (tuple[str, int]): Address of the connecting ground-station
            client (used only for logging).
        drone_ip (str): IPv4 address of the drone to connect to.
        drone_port (int): TCP port on the drone to connect to.
        name (str): Label used in log messages to identify this connection
            (e.g. "COMMAND").

    Returns:
        None: This function does not return a value. Both the ground-side
            connection and the drone-side connection are closed before
            returning (in the nested `relay` helper's finally block).
    '''
    
    logging.info(f'{name} TCP connection from {addr}')
    
    packets = 0 # Packet count for logging
    try:
        drone_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP socket to connect to drone
        drone_sock.settimeout(5.0) # Set timeout for connection attempts
        logging.info(f'{name} attempting to connect to drone at {drone_ip}:{drone_port}...')
        drone_sock.connect((drone_ip, drone_port)) # Connect to drone
        logging.info(f'{name} successfully connected to drone')

        def relay(src, dst, label):
            '''
            Relays data from one TCP socket to another until the source closes,
            a receive times out (10 seconds of inactivity), or an error occurs,
            while running remains True. Logs a running packet count every 100
            forwarded chunks. Always closes both src and dst before returning,
            regardless of how the loop ends.

            Args:
                src (socket.socket): Socket to read data from (4 KiB per recv).
                dst (socket.socket): Socket to write the same data to.
                label (str): Label used in log messages to identify this relay
                    direction.

            Returns:
                None: This function does not return a value; it increments the
                    enclosing scope's `packets` counter (via nonlocal) as a side
                    effect.
            '''
            try:
                src.settimeout(10.0)  # 10-second timeout per receive
                dst.settimeout(10.0)  # 10-second timeout per send
                while running:
                    try:
                        data = src.recv(4096) # Read data from source (4KiB buffer)
                        if not data:
                            logging.info(f'{label} connection closed by peer')
                            break
                        dst.sendall(data) # Send all data to destination
                        nonlocal packets
                        packets += 1
                        if packets % 100 == 0: # Log every 100 packets
                            logging.info(f'{label} forwarded {packets} packets')
                    except socket.timeout:
                        logging.warning(f'{label} receive timeout after 10s inactivity')
                        break
                    
            except Exception as e:
                if running:
                    logging.error(f'{label} relay error: {e}')
                pass
            finally:
                try:
                    src.close()
                except:
                    pass
                try:
                    dst.close()
                except:
                    pass

        # Relay in both directions
        tcp_to_drone = threading.Thread(target=relay, args=(conn, drone_sock, f"{name} <-"), daemon=True) # Laptop to drone
        tcp_to_ground = threading.Thread(target=relay, args=(drone_sock, conn, f"{name} ->"), daemon=True) # Drone to laptop
        
        # Start both relay threads
        tcp_to_drone.start() 
        tcp_to_ground.start()
        tcp_to_drone.join()
        tcp_to_ground.join()

    except Exception as e:
        logging.error(f'{name} TCP connection error: {e}')
    finally:
        try: conn.close()
        except: pass
  
def forward_tcp(listen_port, dest_ip, dest_port, name):
    '''
    Runs a blocking TCP server loop on listen_port (SO_REUSEADDR and
    SO_KEEPALIVE enabled, backlog 5, 1-second accept timeout) that spawns
    a new daemon thread running handle_tcp_connection() for each accepted
    connection, relaying it to (dest_ip, dest_port). Continues accepting
    until the module-level `running` flag becomes False.

    Args:
        listen_port (int): Local TCP port to listen on.
        dest_ip (str): Destination IPv4 address each accepted connection
            is relayed to.
        dest_port (int): Destination TCP port each accepted connection is
            relayed to.
        name (str): Label used in log messages and passed through to
            handle_tcp_connection() (e.g. "COMMAND").

    Returns:
        None: This function does not return a value; it closes the
            listening socket before returning.
    '''
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP socket for listening to laptop connections
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow for address reuse, avoiding restart errors
    server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) # Enable TCP keepalive to detect dead connections
    server.bind(('0.0.0.0', listen_port)) 
    server.listen(5)
    server.settimeout(1.0)

    logging.info(f'{name} TCP: {listen_port} -> {dest_ip}:{dest_port}')

    while running:
        try:
            conn, addr = server.accept() # Wait for a laptop to connect
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) # Enable keepalive on accepted connection
            tcp_handler_thread = threading.Thread(target=handle_tcp_connection, 
            args=(conn, addr, dest_ip, dest_port, name), daemon=True) # Handle connection in separate thread
            tcp_handler_thread.start()
        except socket.timeout:
            continue  # Normal timeout, loop again
        except Exception as e:
            if running:
                logging.error(f'{name} server error: {e}')
    server.close()
    
def main():
    '''
    Entry point for the drone forwarder. Starts three daemon threads --
    one forward_udp() thread each for VIDEO and DATA traffic, and one
    forward_tcp() thread for COMMAND traffic -- all relaying between the
    drone's ad-hoc-network IP and the laptop's IP on their respective
    fixed ports. Blocks the main thread in a 1-second sleep loop until a
    KeyboardInterrupt, at which point the module-level `running` flag is
    cleared, each thread is joined with a 2-second timeout, and the
    process exits.

    Returns:
        None: This function does not return; it calls sys.exit(0) after
            handling a KeyboardInterrupt.
    '''
    global running

    print("Starting drone forwarder...")
    print(f"Forwarding VIDEO UDP: {VIDEO_PORT} -> {LAPTOP_IP}:{VIDEO_PORT}")
    print(f"Forwarding DATA UDP: {DATA_PORT} -> {LAPTOP_IP}:{DATA_PORT}")
    print(f"Forwarding COMMAND TCP: {COMMAND_PORT} -> {DRONE_IP}:{COMMAND_PORT}")
    
    video_thread = threading.Thread(
        target=forward_udp,
        args=(5602, LAPTOP_IP, 5602, 'VIDEO'),
        daemon=True
    )

    data_thread = threading.Thread(
        target=forward_udp,
        args=(5601, LAPTOP_IP, 5601, 'DATA'),
        daemon=True
    )

    command_thread = threading.Thread(
        target=forward_tcp,
        args=(5603, DRONE_IP, 5603, 'COMMAND'),
        daemon=True
    )
    threads = [video_thread, data_thread, command_thread]
    
    for t in threads:
        t.start() # Start all forwarding threads
    
    logging.info('Drone forwarder running. Press Ctrl+C to stop.')
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info('Stopping drone forwarder...')
        running = False
        for t in threads:
            t.join(timeout=2)
        logging.info('Drone forwarder stopped.')
        sys.exit(0)

if __name__ == "__main__":
    main()