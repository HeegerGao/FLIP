import torch
from torch.utils.data import BatchSampler

def padding_flow(flow, obs_history, flow_horizon, pad_history=True, pad_future=True):
    """
    Pad the flow tensor for the first obs_history-1 frames and the last flow_horizon-1 frames.
    """
    history_flow = torch.zeros(obs_history-1, flow[0].shape[0], flow[0].shape[1], flow[0].shape[2]).to(flow.device)
    future_flow = torch.zeros(flow_horizon-1, flow[-1].shape[0], flow[-1].shape[1], flow[-1].shape[2]).to(flow.device)
            
    for t in range(flow_horizon-1):
        a = flow_horizon-t-1
        b = t+1
        future_flow[t][:a] = flow[-1][b:]
        future_flow[t][a:] = flow[-1][-1:].repeat(b, 1, 1)
        
    if not pad_history and not pad_future:
        return flow
    elif not pad_history and pad_future:
        return torch.cat([flow, future_flow], dim=0)
    elif pad_history and not pad_future:
        return torch.cat([history_flow, flow], dim=0)
    elif pad_history and pad_future:
        return torch.cat([history_flow, flow, future_flow], dim=0)

# this enables us to start from a given index and iterate over the dataset
class CustomBatchSampler(BatchSampler):
    def __init__(self, indices, batch_size):
        self.indices = indices
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i + self.batch_size]

    def __len__(self):
        return len(self.indices) // self.batch_size