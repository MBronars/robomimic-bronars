class Objective:
    def __init__(self, name):
        self.name = name
    
    def compute(self, param, **kwargs):
        pass

    def prepare_data(self, obs_dict, goal_dict):
        pass