import torch
from safetensors.torch import save_file, load_file

__all__ = ['save', 'load', 'save_pt', 'load_pt', 'save_safetensors', 'load_safetensors']

def save(model:torch.nn.Module, path:str, format:str='pt')->None:
    if format == 'pt':
        save_pt(model=model, path=path)
    elif format == 'safetensors':
        save_safetensors(model=model, path=path)
    else:
        raise ValueError('''`format` only support `pt` or `safetensors`
                         unavailable format passed''')

def load(model:torch.nn.Module, path:str, format:str='pt')->None:
    if format == 'pt':
        load_pt(model=model, path=path)
    elif format == 'safetensors':
        load_safetensors(model=model, path=path)
    else:
        raise ValueError('''`format` only support `pt` or `safetensors`
                         unavailable format passed''')
    

def save_pt(model:torch.nn.Module, path:str)->None:
    if not path.endswith('.pt'):
        path += '.pt'
    torch.save(model.state_dict(), path)

def load_pt(model:torch.nn.Module, path:str)->None:
    if not path.endswith('.pt'):
        path += '.pt'
    model.load_state_dict(torch.load(path))


def save_safetensors(model:torch.nn.Module, path:str)->None:
    if not path.endswith('.safetensors'):
        path += '.safetensors'
    save_file(model.state_dict(), path)

def load_safetensors(model:torch.nn.Module, path:str)->None:
    if not path.endswith('.safetensors'):
        path += '.safetensors'
    model.load_state_dict(load_file(path))
