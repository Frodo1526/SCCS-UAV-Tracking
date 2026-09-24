import math

# -----------------------------------------------------------------------------------------------
# Tricking Smoother to Reduce Jitter (not implemented in main)
# -----------------------------------------------------------------------------------------------
# Implements scaling and deadzone to reduze the movement when a target is relatively centered.
# Helps with flight stability, also reduces the num of commands required, saving a lot of overhead.
class JitterReducer:
    def __init__(self, dead_zone_radius=10, smoothing_factor=0.8):
        '''
        Initializes a JitterReducer with a dead-zone radius and an exponential
        smoothing factor, and no prior position recorded yet.

        Args:
            dead_zone_radius (float): Radius in pixels around a reference
                center point within which a position is snapped to the center
                (see is_in_dead_zone / reduce_jitter). Defaults to 10.
            smoothing_factor (float): Weight given to the current position in
                the exponential moving average used by smooth_position(); 1.0
                would apply no smoothing, lower values smooth more heavily.
                Defaults to 0.8.

        Returns:
            None: This function does not return a value.
        '''
        # Initialize the jitter reducer.
        self.dead_zone_radius = dead_zone_radius
        self.smoothing_factor = smoothing_factor
        self.previous_position = None

    def is_in_dead_zone(self, x, y, center_x, center_y):
        '''
        Determines whether a point lies within dead_zone_radius pixels of a
        given center point.

        Args:
            x (float): X-coordinate of the point to test.
            y (float): Y-coordinate of the point to test.
            center_x (float): X-coordinate of the reference center.
            center_y (float): Y-coordinate of the reference center.

        Returns:
            bool: True if the Euclidean distance from (x, y) to
                (center_x, center_y) is less than self.dead_zone_radius,
                False otherwise.
        '''
        # Check target is within dead zone.
        distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        return distance < self.dead_zone_radius

    def smooth_position(self, current_position):
        '''
        Smooths a position using an exponential moving average against the
        previously smoothed position, to reduce frame-to-frame jitter.

        Args:
            current_position (tuple[float, float]): The raw (x, y) position
                for the current frame.

        Returns:
            tuple[float, float]: current_position unchanged, if this is the
                first call (no previous position recorded yet); this call also
                seeds self.previous_position with current_position.
            tuple[float, float]: (smoothed_x, smoothed_y), each computed as
                smoothing_factor * current + (1 - smoothing_factor) *
                previous, on subsequent calls. self.previous_position is
                updated to this smoothed value.
        '''
        # Smooth the position using an exponential moving average.
        if self.previous_position is None:
            self.previous_position = current_position
            return current_position

        smoothed_x = (
            self.smoothing_factor * current_position[0]
            + (1 - self.smoothing_factor) * self.previous_position[0]
        )
        smoothed_y = (
            self.smoothing_factor * current_position[1]
            + (1 - self.smoothing_factor) * self.previous_position[1]
        )

        self.previous_position = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def reduce_jitter(self, x, y, center_x, center_y):
        '''
        Combines dead-zone snapping and exponential smoothing to produce a
        jitter-reduced position: positions within the dead zone are snapped
        exactly to the center; positions outside it are exponentially
        smoothed via smooth_position().

        Args:
            x (float): X-coordinate of the raw position.
            y (float): Y-coordinate of the raw position.
            center_x (float): X-coordinate of the reference center.
            center_y (float): Y-coordinate of the reference center.

        Returns:
            tuple[float, float]: (center_x, center_y) if the input position is
                within the dead zone (per is_in_dead_zone).
            tuple[float, float]: The smoothed (x, y) position from
                smooth_position() otherwise.
        '''
            # Reduce jitter for the given position.
        if self.is_in_dead_zone(x, y, center_x, center_y):
            # If in the dead zone, return the center position
            return center_x, center_y
        else:
            # Smooth the position
            return self.smooth_position((x, y))