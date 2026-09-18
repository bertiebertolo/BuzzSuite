class mosquito_obj_tracker:
    r"""Placeholder for unpickling BuzzWatch tracking objects when the real
    buzzwatch_data_analysis package is unavailable. Pickle instantiates via __new__
    and populates __dict__ (time_stamp, objects, ...) directly, so no real
    behaviour is required for read-only trajectory analysis."""
    def __setstate__(self, state):
        self.__dict__.update(state)
