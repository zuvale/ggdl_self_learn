from mealpy import Problem, FloatVar, IntegerVar, StringVar, MixedSetVar
from mealpy.utils.space import BaseVar
from numpy import ndarray
import torch
import torch.nn as nn
from typing import Dict, Iterable, Tuple
from generative.base_dists import GaussianBase
from generative.diffusion import (
    ConstantLinearSchedule, LearnableLinearSchedule,
    DDPMReverseProcess, DDIMReverseProcess
)
from .basic_nn import create_unet
from .utils import get_act_fun, get_optim


DIFFUSION_BOUNDS = [
    IntegerVar(lb=12, ub=1024, name="time_emb_size"),
    IntegerVar(lb=12, ub=1024, name="class_emb_size"),
    StringVar(
        valid_sets=(
            "relu", "relu6", "prelu", "rrelu", "selu", "silu", "gelu"),
        name="f_act_fun"
    )
]


class DiffusionSearchProblem(Problem):
    def __init__(
        self, bounds: Iterable[BaseVar]|None=None, minmax: str="min",
        data: None=None, data_shape: Tuple[int]=(1,), train_loader=None,
        n_epochs: int=100, n_timesteps: int=1000, device: str="cpu", **kwargs
    ) -> None:
        self.data_shape = data_shape
        self.train_loader = train_loader
        self.n_epochs = n_epochs
        self.n_timesteps = n_timesteps
        self.device = device
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x: ndarray) -> float:
        x_decoded = self.decode_solution(x)

        try:
            model = DDPMReverseProcess(
                GaussianBase(self.data_shape),
                ConstantLinearSchedule(self.n_timesteps),
                self._define_score_network(x_decoded),
                self.n_timesteps
            ).to(self.device)

            optim_fun = get_optim(x_decoded["optim_fun"])
            optimizer = optim_fun(
                model.parameters(), lr=x_decoded["learning_rate"])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=x_decoded["gamma"])
            
            self.training_loop(model, optimizer, scheduler)

            return self.evaluate(model)

        except:
            print("bad initialization!")
            return 1000000.

    def training_loop(self, model: nn.Module, optimizer, scheduler) -> None:
        return NotImplementedError
    
    def evaluate(self, model) -> float:
        return NotImplementedError
    
    def _define_score_network(self, hyperpars: Dict) -> nn.Module:
        return NotImplementedError

class DiffusionMNISTSearchProblem(DiffusionSearchProblem):
    N_CLASSES: int=10
    def training_loop(self, model: nn.Module, optimizer, scheduler) -> None:
        model.train()
        for epoch in range(self.n_epochs):
            for x, y in self.train_loader:
                optimizer.zero_grad()

                x = x.to(self.device)
                loss = model(x, y=y)

                loss.backward()
                optimizer.step()
            
            scheduler.step()

class DDPMUNetMNISTSearchProblem(DiffusionMNISTSearchProblem):
    def _define_score_network(self, hyperpars: Dict) -> nn.Module:
        return create_unet(
            1, self.data_shape[-1], [
                hyperpars["hid_chan_size_1"], hyperpars["hid_chan_size_2"],
                hyperpars["hid_chan_size_3"], hyperpars["hid_chan_size_4"],
                hyperpars["hid_chan_size_5"], hyperpars["hid_chan_size_6"],
                hyperpars["hid_chan_size_7"]
            ], hyperpars["n_hidden_layers"], hyperpars["conv_kernel_size"],
            hyperpars["pool_kernel_size"], hyperpars["pool_stride_size"],
            [
                hyperpars["pool_pad_size_1"], hyperpars["pool_pad_size_2"],
                hyperpars["pool_pad_size_3"], hyperpars["pool_pad_size_4"],
                hyperpars["pool_pad_size_5"], hyperpars["pool_pad_size_6"],
                hyperpars["pool_pad_size_7"]
            ], hyperpars["tconv_kernel_size"],
            self.n_timesteps, hyperpars["time_emb_size"],
            hyperpars["class_emb_size"], self.N_CLASSES,
            activation_function=get_act_fun(hyperpars["u_act_fun"]),
            unet_type="time_class_filmed",
            film_act_fun=get_act_fun(hyperpars["f_act_fun"])
        )