from hyperspy._signals.image import Image
import hyperspy.misc.image_peak_finder

class HighResADF(Image):
    _signal_type = "HighResADF"

    def __init__(self, *args, **kwards):
        Image.__init__(self, *args, **kwards)
        # Attributes defaults
        self.metadata.Signal.binned = True


    def peak_find_ranger(self,
                         best_size = "auto",
                         refine_positions=False,
                         show_progress=False,
                         sensitivity_threshold=0.34,
                         start_search=3,
                         end_search="auto"):
        
        """
        
        Parameters
        ----------
        refine_position : bool
            ddf
            
        """
        return hyperspy.misc.image_peak_finder(self.data, best_size, refine_position,
                                               show_progress, sensitivity_threshold,
                                               start_search, end_search)
