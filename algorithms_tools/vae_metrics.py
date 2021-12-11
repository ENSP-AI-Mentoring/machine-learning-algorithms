import torch

def marginals_anderson_darling_torch(data: torch.Tensor, generated: torch.Tensor):
    scores = 0
    
    for k in range(data.shape[1]):
        data_i = data[:, k]
        generated_i, _ = torch.sort(generated[:, k])
        
        n = len(data_i)
        
        concordances = torch.tensor([
            torch.sum(
                data_i <= g
            )+1 for g in generated_i
        ])
        
        concordances = concordances / (n+2)
        
        error = -n - (1/n) * torch.sum(torch.tensor([
            (2*i-1) * (torch.log(concordances[i-1]) + torch.log(1-concordances[n - i])) for i in range(1, n+1)
        ]))
        
        scores += error
    return scores/data.shape[1]

def absolute_kendall_error_torch(data: torch.Tensor, generated: torch.Tensor):
    res = 0
    n = data.shape[0] # 410 or 408
    d = data.shape[1] # 4
    
    for i in range(n):
        data_i = torch.concat([data[:i], data[(i+1):]])
        generated_i = torch.concat([generated[:i], generated[(i+1):]])
        
        Zi = torch.mean((torch.sum(data_i < data[i], axis=1)==d).float())
        Zti = torch.mean((torch.sum(generated_i < generated[i], axis=1)==d).float())
        
        res += torch.abs(Zti - Zi)
    return res/n