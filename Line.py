import numpy as np


class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # polynomial coefficients averaged over the last n iterations
        self.last_fits = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
