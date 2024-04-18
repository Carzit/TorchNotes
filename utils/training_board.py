import os
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tqdm import tqdm
from save_and_load import save

def train(
        epoches:int, 
        optimizer:torch.optim.Optimizer,
        lr_scheduler:torch.optim.lr_scheduler.LRScheduler,
        model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        train_generator:DataLoader, 
        val_generator:DataLoader,
        *,
        log_dir:str = r".\log",
        print_per_epoch:int=1,
        save_per_epoch:int=1,
        save_path:str=os.curdir,
        save_name:str="model",
        save_format:str="pt",
        device:torch.device=torch.device('cpu'))->torch.nn.Module:

    writer = SummaryWriter(os.path.join(log_dir, "TRAIN"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    model = model.to(device=device)
    

    for epoch in range(epoches):
        
        # Train
        model.train()
        for batch, (X, Y) in enumerate(tqdm(train_generator)):

            X = X.to(device=device)
            Y = Y.to(device=device)

            output = model(X)
            train_loss = loss_fn(output, Y)  

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        writer.add_scalar("Train Loss", train_loss.item(), epoch)
        
        # If validation datasets exisit, calculate val loss without recording grad.
        if val_generator:
            model.eval()
            with torch.no_grad():
                for batch, (X, Y) in enumerate(val_generator):
                    X = X.to(device=device)
                    Y = Y.to(device=device)
                    output = model(X)
                    val_loss = loss_fn(output, Y)
                
                writer.add_scalar("Validation Loss", val_loss.item(), epoch)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss.item(), "Validation Loss": val_loss.item()}, epoch)

        # If learning rate scheduler exisit, update learning rate per epoch.
        if lr_scheduler:
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)
            lr_scheduler.step()
        
        # Flushes the event file to disk
        writer.flush()

        # Specify print_per_epoch = 0 to unable print training information.
        if print_per_epoch:
            if (epoch+1) % print_per_epoch == 0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, epoches, train_loss.item(), val_loss.item()))
        
        # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
        if save_per_epoch:
            if (epoch+1) % save_per_epoch == 0:
                model_name = f"{save_name}_epoch{epoch}"
                model_path = os.path.join(save_path, model_name)
                print(model_path)
                save(model, model_path, save_format)
        
        
    writer.close()
    model_name = f"{save_name}_final"
    model_path = os.path.join(save_path, model_name)
    save(model, model_path, save_format)
    
    return model
    
def test(model:torch.nn.Module, 
        loss_fn:torch.nn.Module, 
        test_generator:DataLoader,
        *,
        log_dir:str = r".\log",
        device:torch.device=torch.device('cpu'))->None:
    
    writer = SummaryWriter(os.path.join(log_dir, "TEST"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    model = model.to(device=device)
    model.eval()

    total_inaccuracy = 0
    total_batch = 0
    for batch, (X, Y) in enumerate(test_generator):
        X = X.to(device=device)
        Y = Y.to(device=device)

        output = model(X)

        test_inaccuracy = loss_fn(output, Y)
        total_inaccuracy += test_inaccuracy.item()
        total_batch += 1

        writer.add_scalar("Criterion per Batch", test_inaccuracy.item(), batch)
        writer.add_scalar("Criterion Average", total_inaccuracy/total_batch, batch)

    print('Test Inaccuracy: {:.4f}'.format(total_inaccuracy/total_batch))