    
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def to_onehot(a, num_classes=22):
  b = torch.zeros((a.shape[0], num_classes))
  b[torch.arange(a.shape[0]), a-1] = 1
  return b

class yoga_loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.L1Loss = nn.L1Loss(reduction='mean')
    
  def mpjpe(self, predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))
    
  def forward(self, inp):
    out_corr, pose, corr_pose, label = inp

    pred_dist = F.cross_entropy(out_corr, label)
    # print(pred_dist)
    pose_dist = self.mpjpe(corr_pose, pose)
    pose_dist_l1 = self.L1Loss(corr_pose, pose)
    return 0.35*pred_dist + 0.65*(18*pose_dist_l1), pred_dist, pose_dist 
    