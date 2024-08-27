import abc

class Constraint:
    def __init__(self, name):
        self.name = name
    
    @abc.abstractclassmethod
    def __call__(self, param, **kwargs):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def prepare(self, obs_dict):
        raise NotImplementedError

    def test(self):
        """
        Test if the implemented jacobian is valid.
        """
        raise NotImplementedError

class ArmCollisionConstraint(Constraint):
    def __init__(self):
        super().__init__(name = "arm_collision")

    def prepare(self, obs_dict):
        pass

    def __call__(self, param, **kwargs):
        pass

class GripperCollisionConstraint(Constraint):
    def __init__(self):
        super().__init__(name = "gripper_collision")

    def prepare(self, obs_dict):
        pass

    def __call__(self, param, **kwargs):
        pass

class JointConstraint(Constraint):
    def __init__(self):
        super().__init__(name = "joint")

    def prepare(self, obs_dict):
        pass

    def __call__(self, param, **kwargs):
        pass

