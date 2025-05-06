# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
import math

class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer. 
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.
    
    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.
    
    Arguments:
        params (iterable): 
            Iterable of parameters to optimize or dicts defining 
            parameter groups.
        lr (float): 
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): 
            Term added to the denominator outside of the root operation to 
            improve numerical stability. (default: 1e-8).
        weight_decay (float): 
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """
    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 warmup_steps: int = 0,
                 stochastic_rounding=True,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = False
                 ):

        defaults = dict(lr=lr, 
                        betas=betas, 
                        eps=eps,
                        r=r,
                        k=0,
                        warmup_steps=warmup_steps,
                        lr_prev=-1.0,
                        train_mode=True,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        stochastic_rounding=stochastic_rounding,
                        foreach=foreach)
        super().__init__(params, defaults)
        print("AdamW Schedule-Free optimizer initialized. PDP1.9.1")
        self.fused_back_pass = False
        self.try_hook_kohya_fbp()

    def try_hook_kohya_fbp(self):
        self.kohya_original_patch_adafactor_fused = None

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # Get the original method so we can restore it later.
            self.kohya_original_patch_adafactor_fused = library.adafactor_fused.patch_adafactor_fused

            # Define the override.
            def prodigy_patch_adafactor_fused(optimizer):
                unwrapped_optimiser = None
                if hasattr(optimizer, "optimizer"):
                    # If the optimiser is wrapped, forward the calls to the actual optimiser.
                    def _step(self, *args, **kwargs):
                        return self.optimizer.step(*args, **kwargs)

                    def _step_param(self, *args, **kwargs):
                        return self.optimizer.step_param(*args, **kwargs)

                    optimizer.step = _step.__get__(optimizer)
                    optimizer.step_param = _step_param.__get__(optimizer)
                    unwrapped_optimiser = optimizer.optimizer
                else:
                    unwrapped_optimiser = optimizer
               
                print(f"[{self.__class__.__name__}] Kohya pipeline detected with fused backward pass. Gradient hook patch successful.")
                library.adafactor_fused.patch_adafactor_fused = unwrapped_optimiser.kohya_original_patch_adafactor_fused # Restore the original method.

                unwrapped_optimiser.fused_back_pass = True
                unwrapped_optimiser.kohya_original_patch_adafactor_fused = None

            # Patch the method.
            library.adafactor_fused.patch_adafactor_fused = prodigy_patch_adafactor_fused
        except:
            pass

    def try_unhook_kohya_fbp(self):
        if self.kohya_original_patch_adafactor_fused is None:
            return

        try:
            # Import and patching will fail if not Kohya.
            import library.adafactor_fused

            # User did not opt for fused backward pass, so remove our hook.
            library.adafactor_fused.patch_adafactor_fused = self.kohya_original_patch_adafactor_fused
        except:
            pass

        self.kohya_original_patch_adafactor_fused = None

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if train_mode:
                for p in group['params']:
                    z = self.state[p].get('z')
                    if z is not None:
                        # Set p to x
                        p.lerp_(end=z.to(device=p.device), weight=1 - 1 / beta1)
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1, _ = group['betas']
            if not train_mode:
                for p in group['params']:
                    z = self.state[p].get('z')
                    if z is not None:
                        # Set p to y
                        p.lerp_(end=z.to(device=p.device), weight=1 - beta1)
                group['train_mode'] = True

    # Implementation by Nerogar. From: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    def copy_stochastic_(self, target, source):
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

    def smart_copy(self, target, source, stochastic_rounding, smart_delete_source):
        if target is source:
            return

        if stochastic_rounding and target.dtype == torch.bfloat16 and source.dtype == torch.float32:
            self.copy_stochastic_(target, source)
        else:
            target.copy_(source)

        if smart_delete_source:
            del source

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.try_unhook_kohya_fbp()

        if self.fused_back_pass:
            return

        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            eps = group['eps']
            beta1, beta2 = group['betas']
            decay = group['weight_decay']
            k = group['k']
            r = group['r']
            warmup_steps = group['warmup_steps']
            weight_lr_power = group['weight_lr_power']
            
            if k < warmup_steps:
              sched = (k+1) / warmup_steps
            else:
              sched = 1.0
            
            bias_correction2 = 1
            lr = group['lr']*sched*math.sqrt(bias_correction2)
            
            if group['lr_prev'] == -1.0:
                d_k = group['d_k'] = 1
            else:
                d_k = group['d_k'] =  (group['lr_prev'] / lr)
            group['lr_prev'] = lr


            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]
            
            for p in active_p:
                if 'z' not in self.state[p]:
                    self.state[p]['z'] = p.detach().clone(memory_format=torch.preserve_format)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format).detach()

            for p in active_p:
                state = self.state[p]
                z_state = state['z']
                stochastic = group['stochastic_rounding']

                y, z = (p.float(), z_state.float()) if stochastic else (p, z_state)
                grad = p.grad.to(dtype=torch.float32, copy=True)
                

                update = None
                
                
                weight_sum = group['weight_sum']

                #update_second_moment
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(beta2 * d_k * d_k).addcmul_(grad, grad, value=1-beta2)
                denom = exp_avg_sq.sqrt()

                #update_
                update = grad.div_(denom.add_(eps))
                del denom
                
                if update is not None:
                    #update_params
                    weight = lr ** 2
                    weight_sum = group['weight_sum'] + weight
                    ckp1 = weight / weight_sum if weight_sum else 0

                    xy_step = 1 - beta1 * (1 - ckp1)

                    y.lerp_(end=z, weight=ckp1)

                    if decay != 0:
                        decay *= lr
                        z.sub_(y, alpha=decay)
                        y.sub_(y, alpha=decay * xy_step)

                    z.sub_(update, alpha=lr)
                    y.sub_(update, alpha=lr * xy_step)

                    self.smart_copy(p, y, stochastic, True)
                    self.smart_copy(z_state, z, stochastic, True)

                    del update

            group['weight_sum'] = weight_sum
            group['k'] = k+1
        return loss
