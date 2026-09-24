import pygame
import logging

from data_cmd import send_command_to_air_pi

def select_target_by_click(click_pos, tracking_data, app_state):
    '''
    Determines whether a screen click landed inside any currently tracked
    object's bounding box, updates app_state.current_target_id to that
    object's ID (or to None if the click missed every box), and forwards
    the selection to the Air Pi via send_command_to_air_pi().

    Iterates tracking_data in order and selects the first bounding box
    that contains click_pos; later overlapping boxes are not considered
    once a match is found.

    Args:
        click_pos (tuple[int, int]): (x, y) pixel coordinates of the mouse
            click, in the same coordinate space as the bounding boxes in
            tracking_data.
        tracking_data (list[dict]): Current tracked-object list, each dict
            expected to contain 'bbox' (start_x, start_y, end_x, end_y)
            and 'id'.
        app_state (AppState): Shared application state; current_target_id
            is updated under app_state.target_lock.

    Returns:
        None: This function does not return a value; it updates
            app_state.current_target_id and sends a command over the
            network as side effects.
    '''
    found_target = False
    new_target_id = None
   
    for data in tracking_data:
        # Setting starting and ending coordinates of bounding box from tracking data
        (start_x, start_y, end_x, end_y) = data['bbox']
        # If the click position is within the bounding box, we have found a target
        if (start_x) < click_pos[0] < (end_x) and (start_y) < click_pos[1] < (end_y):
            new_target_id = data['id']
            logging.info(f"Click at {click_pos} matched target ID {new_target_id} with bbox {data['bbox']}")
            found_target = True
            break
           
    with app_state.target_lock:
        app_state.current_target_id = new_target_id
       
    if found_target:
        logging.info(f"Clicked new target: ID {new_target_id}")
    else:
        logging.info("Clicked empty space, clearing target.")
       
    # Not really sending directly to Air Pi, maybe change name?
    send_command_to_air_pi({'target_id': new_target_id})
    
# -----------------------------------------------------------------------------------------------
# Pygame UI Functions
# -----------------------------------------------------------------------------------------------
def handle_input(events, ui_state, app_state, tracking_data):
    '''
    Processes a batch of Pygame events for the ground station UI: quits on
    a window-close event or the Q/Escape key, selects/deselects a target
    on left mouse click (via select_target_by_click), and clears the
    current target when the C key is pressed (updating app_state directly
    and notifying the Air Pi).

    Args:
        events (list[pygame.event.Event]): Events to process, typically
            from pygame.event.get().
        ui_state (dict): Mutable UI state dict; this function may set
            ui_state['running'] = False to signal the main loop to exit.
        app_state (AppState): Shared application state used/updated for
            target selection/clearing.
        tracking_data (list[dict]): Current tracked-object list, passed
            through to select_target_by_click() for click hit-testing.

    Returns:
        None: This function does not return a value.
    '''
    for event in events:
        if event.type == pygame.QUIT:
            ui_state['running'] = False
           
        elif event.type == pygame.MOUSEBUTTONDOWN:
            click_pos = pygame.mouse.get_pos()
            # Handle click in main thread to avoid race conditions
            select_target_by_click(click_pos, tracking_data, app_state)
               
        elif event.type == pygame.KEYDOWN:
            # Clear target
            if event.key == pygame.K_c:
                logging.info("Target cleared by 'C' key")
                with app_state.target_lock:
                    app_state.current_target_id = None
                send_command_to_air_pi({'target_id': None})
            # Quit
            elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                ui_state['running'] = False


def render_graphics(screen, frame, tracking_data, current_target_id, ui_state, fonts):
    '''
    Renders one frame of the ground station UI: either a "waiting for
    video" placeholder screen (if no frame is available yet), or the
    latest video frame with bounding-box overlays for every tracked
    object (red/thicker for the current target, green for others, each
    labeled with its ID, class label, and a star marker plus centroid
    cross-hair for the target), plus a fixed instruction line at the
    bottom of the screen. Always ends by calling pygame.display.flip().

    Args:
        screen (pygame.Surface): The window surface to draw into.
        frame (numpy.ndarray | None): The latest decoded RGB video frame
            (rotated per GStreamer's layout, transposed here to draw
            upright), or None if no frame has arrived yet.
        tracking_data (list[dict]): Current tracked-object list, each dict
            expected to contain 'bbox', 'id', and 'label'.
        current_target_id: The ID of the currently selected target, used
            to pick highlight color/label styling, or None/absent if no
            target is selected.
        ui_state (dict): UI state dict; not read or written by this
            function directly (accepted for a consistent call signature
            with the rest of the render/input pipeline).
        fonts (dict): Mapping with at least 'main' and 'small' pygame.font
            Font objects, used for the placeholder text and for object
            labels/instructions respectively.

    Returns:
        None: This function does not return a value; it draws to screen
            and flips the display as a side effect. If frame is None, it
            returns early after drawing the placeholder screen.
    '''
    if not tracking_data:
        logging.debug("No tracking data to render.")
    else:
        logging.debug(f"Rendering {len(tracking_data)} tracked objects. Current target ID: {current_target_id}")
   
    if frame is None:
        # Display "Waiting for video" message
        screen.fill((30, 30, 30)) # Dark grey background
        wait_text = "Waiting for video stream... (Radxa USB connected?)"
        text_surface = fonts['main'].render(wait_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=screen.get_rect().center)
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        return

    # Convert numpy array to pygame surface
    # The frame is already RGB, but GStreamer gives it to us rotated.
    # We must rotate it and flip it to be upright.
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(frame_surface, (0, 0))
    frame_width, frame_height = screen.get_size()

    # Draw tracked objects
    for data in tracking_data:
        (start_x, start_y, end_x, end_y) = data['bbox']
        is_target = (data['id'] == current_target_id)
       
        # Red for target, green for others
        color = (255, 0, 0) if is_target else (0, 255, 0)
        thickness = 3 if is_target else 2
       
        # Draw bounding box
        pygame.draw.rect(screen, color, (start_x, start_y, end_x - start_x, end_y - start_y), thickness)
       
        # Draw label
        label_text = f"ID:{data['id']} {data['label']}"
        if is_target:
            label_text = f"★ {label_text} ★"
        text_surface = fonts['small'].render(label_text, True, color, (0,0,0))
        screen.blit(text_surface, (start_x, start_y - 25))
   
    # Draw instructions
    instructions = ["Click object to track | C: Clear target | Q/ESC: Quit"]
    y_offset = frame_height - 30
    for instruction in instructions:
        instr_surface = fonts['small'].render(instruction, True, (200, 200, 200), (0, 0, 0))
        screen.blit(instr_surface, (10, y_offset))
        y_offset += 25
   
    pygame.display.flip()