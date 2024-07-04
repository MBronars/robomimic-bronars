import torch


# Plan Data Structure
class ReferenceTrajectory:
    # TODO: change the dtype and device
    # TODO: change the code accordingly for the safety filter
    def __init__(self, t_des, x_des, dx_des=None, dtype=torch.float32, device=torch.device("cpu")):
        assert t_des.shape[0] == x_des.shape[0]

        if dx_des is None:
            # If dx_des is not provided, compute it from x_des using forward difference
            dt_des = (t_des[1:] - t_des[:-1]).unsqueeze(-1)
            dx_des = (x_des[1:] - x_des[:-1])/dt_des
            x_des = x_des[:-1]
            t_des = t_des[:-1]
        else:
            assert t_des.shape[0] == dx_des.shape[0]
            x_des = x_des
            t_des = t_des
            dx_des = dx_des
        
        self.dtype = dtype
        self.device = device
        
        self.t_des = self.to_tensor(t_des)
        self.x_des = self.to_tensor(x_des)
        self.dx_des = self.to_tensor(dx_des)

        self._created_from_traj_param = False
    
    def to_tensor(self, x):
        return torch.tensor(x, dtype=self.dtype, device=self.device)


    def __getitem__(self, key):
        """ 
        Returns another ReferenceTrajectory object with the sliced data
        """
        if isinstance(key, slice):
            plan = ReferenceTrajectory(self.t_des[key], self.x_des[key], self.dx_des[key])
            if self._created_from_traj_param:
                plan.stamp_trajectory_parameter(self._traj_param)
            return plan
        
        elif isinstance(key, int):
            if key < 0:
                key = len(self) + key
            if key >= self.__len__():
                raise IndexError("Index out of range")
            return (self.t_des[key], self.x_des[key], self.dx_des[key])
        
        elif isinstance(key, (list, torch.Tensor)) and key.dtype == torch.bool:
            # Handle boolean array masking
            if len(key) != len(self.t_des):
                raise ValueError("Boolean mask must have the same length as the reference trajectory")
            
            plan = ReferenceTrajectory(self.t_des[key], self.x_des[key], self.dx_des[key])
            if self._created_from_traj_param:
                plan.stamp_trajectory_parameter(self._traj_param)
            return plan

        else:
            raise TypeError("Invalid argument type")
        
    def __len__(self):
        return self.t_des.shape[0]
    
    def set_start_time(self, t_start = 0):
        """
        Shift the time vector to start from t_start
        """
        self.t_des = self.t_des - self.t_des[0] + t_start

    def stamp_trajectory_parameter(self, traj_param):
        """
        Stamp the parameter k to the reference trajectory
        """
        self._created_from_traj_param = True
        self._traj_param = traj_param
    
    def get_trajectory_parameter(self):
        """
        Get the stamped parameter to the reference trajectory
        """
        assert self._created_from_traj_param
        return self._traj_param