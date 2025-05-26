import torch

def tensor_min_max(t):
    tt_min,_ = torch.min(t, axis= 0)
    tt_max,_ = torch.max(t, axis= 0)
    return torch.stack((tt_min, tt_max), dim=1)

def final_tensor(tensor1,tensor2):
    if torch.isnan(tensor1).all().item()==True :
        t1 = tensor_min_max(tensor2)
        t2 = tensor_min_max(tensor2)
    # elif torch.isnan(tensor2).all().item()==True:
    #     t1 = tensor_min_max(tensor1)
    #     t2 = tensor_min_max(tensor1)
    else:
        t1 = tensor_min_max(tensor1)
        t2 = tensor_min_max(tensor2)
    
    min_tensor = torch.min(t1, t2)
    max_tensor = torch.max(t1, t2)
    
    return torch.stack((min_tensor[:,0],max_tensor[:,1]), axis=1)

def comparaison(tensor1, tensor2):
    if torch.isnan(tensor1).all().item()==True :
        return tensor2
    else :
        return torch.stack((torch.min(tensor1, tensor2)[:,0], torch.max(tensor1, tensor2)[:,1]), axis = 1)