from mealpy import Problem, FloatVar, IntegerVar, StringVar, MixedSetVar
from mealpy.utils.space import BaseVar
from numpy import ndarray
import rdkit.Chem as Chem
from rdkit import rdBase
import torch
import torch.nn as nn
from typing import Dict, Iterable, Tuple

from data_proc.mol_preproc import (
    TU_MUTAG_CONFIG, batch_to_mols, fix_nitro_charges)
from gnn_.architectures import GNNDenoiser
from generative.base_dists import GaussianBase, MarginalGraphBase
from generative.diffusion import (
    ConstantLinearSchedule, LearnableLinearSchedule,
    DDPMReverseProcess, VDMReverseProcess
)
from generative.discrete_flow_matching import (
    LinearDiscreteNoiser, LinearDiscreteRateMatrix, CTMCSampler, DeFoG
)
from generative.utils import LinearMonotonicNetwork
from utilities.metrics import DiffusionBPDEvaluator
from .basic_nn import create_unet, POWERS_OF_2
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

VDM_DIFFUSION_BOUNDS = DIFFUSION_BOUNDS + [
    FloatVar(lb=-30, ub=-0.1, name="gamma_min"),
    FloatVar(lb=0.1, ub=30, name="gamma_max"),
    MixedSetVar(valid_sets=POWERS_OF_2, name="gamma_hid_size")
]

DEFOG_BOUNDS = [
    MixedSetVar(valid_sets=[32, 48, 64, 96], name="node_hidden"),
    MixedSetVar(valid_sets=[32, 48, 64, 96], name="edge_hidden"),
    MixedSetVar(valid_sets=[32, 48, 64, 96], name="time_emb_size"),
    MixedSetVar(valid_sets=[32, 48, 64, 96], name="class_emb_size"),
    IntegerVar(lb=1, ub=5, name="n_layers"),
    FloatVar(lb=0.5, ub=4.0, name="lambda_eloss"),
    FloatVar(lb=0.3, ub=1.0, name="dbl_weight"),
    FloatVar(lb=0.05, ub=0.5, name="tpl_weight"),
    FloatVar(lb=0.8, ub=2.0, name="ne_weight"),
    FloatVar(lb=-1.5, ub=-0.5, name="dbl_bias"),
    FloatVar(lb=-4.0, ub=-1.0, name="tpl_bias"),
    FloatVar(lb=0.0, ub=1.0, name="ne_bias"),
    FloatVar(lb=0.2, ub=0.8, name="rate_exit_cap"),
    FloatVar(lb=0.5, ub=1.0, name="node_scale"),
    FloatVar(lb=0.3, ub=1.0, name="edge_scale"),
]


class DiffusionSearchProblem(Problem):
    def __init__(
        self, bounds: Iterable[BaseVar]|None=None, minmax: str="min",
        data: None=None, data_shape: Tuple[int]=(1,), train_loader=None,
        test_loader=None, n_epochs: int=100, n_timesteps: int=1000,
        device: str="cpu", **kwargs
    ) -> None:
        self.data_shape = data_shape
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.n_timesteps = n_timesteps
        self.device = device
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x: ndarray) -> float:
        x_decoded = self.decode_solution(x)

        try:
            model = self._define_diffusion_process(
                self._define_score_network(x_decoded), x_decoded).to(self.device)

            optim_fun = get_optim(x_decoded["optim_fun"])
            optimizer = optim_fun(
                model.parameters(), lr=x_decoded["learning_rate"])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=x_decoded["gamma"])
            
            self.training_loop(model, optimizer, scheduler)

        except:
            print("bad initialization!")
            return 1000000.

        obj_val = self.evaluate(model)
        print(f"objective value: {obj_val:.4f}")

        # clean up the GPU memory since the model is not needed anymore
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache() 

        return obj_val

    def training_loop(self, model: nn.Module, optimizer, scheduler) -> None:
        return NotImplementedError
    
    def evaluate(self, model: nn.Module) -> float:
        return NotImplementedError

    def _define_diffusion_process(
        self, score_net: nn.Module, hyperpars: Dict) -> nn.Module:
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
    def _define_diffusion_process(
        self, score_net: nn.Module, hyperpars: Dict) -> nn.Module:
        return DDPMReverseProcess(
            GaussianBase(self.data_shape),
            ConstantLinearSchedule(self.n_timesteps),
            score_net,
            self.n_timesteps
        )

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
    
    def evaluate(self, model: nn.Module) -> float:
        model.eval()
        evaluator = DiffusionBPDEvaluator(model)

        bpd = 0
        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            x = x*2.0 - 1.0

            metrics = evaluator.calculate_bpd(x, y=y)
            bpd_i = metrics["total_bpd"].mean().detach().cpu().numpy().item()
            bpd += bpd_i
        
        bpd /= len(self.test_loader)
        return bpd

class VDMUNetMNISTSearchProblem(DDPMUNetMNISTSearchProblem):
    def _define_diffusion_process(
        self, score_net: nn.Module, hyperpars: Dict) -> nn.Module:
        return VDMReverseProcess(
            GaussianBase(self.data_shape),
            LearnableLinearSchedule(
                self.n_timesteps,
                LinearMonotonicNetwork(1, hyperpars["gamma_hid_size"]),
                gamma_limits=(hyperpars["gamma_min"], hyperpars["gamma_max"])
            ),
            score_net,
            self.n_timesteps
        )

class DeFoGMolSearchProblem(Problem):
    def __init__(
        self, bounds: Iterable[BaseVar]|None=None, minmax: str="min",
        data_set=None, data_loader=None, n_epochs: int=100,
        n_solv_steps: int=100, n_samples: int=100, device: str="cpu",
        **kwargs
    ) -> None:
        self.data_set = data_set
        self.data_loader = data_loader
        self.n_epochs = n_epochs
        self.n_solv_steps = n_solv_steps
        self.n_samples = n_samples
        self.device = device
        super().__init__(bounds, minmax, **kwargs)
    
    def obj_func(self, x: ndarray) -> float:
        x_decoded = self.decode_solution(x)
        
        base_dist = MarginalGraphBase(self.data_set, device=self.device)
        noiser = LinearDiscreteNoiser(base_dist)
        
        base_dist = MarginalGraphBase(self.data_set, device=self.device)
        noiser = LinearDiscreteNoiser(base_dist)
        n_atom_tokens, n_bond_tokens = 8, 4
        node_hidden_size, edge_hidden_size = x_decoded["node_hidden"], x_decoded["edge_hidden"]
        time_emb_size, class_emb_size = x_decoded["time_emb_size"], x_decoded["class_emb_size"]
        n_classes = 2
        n_update = x_decoded["n_layers"]
        denoiser = GNNDenoiser(
            n_atom_tokens, n_bond_tokens, node_hidden_size, edge_hidden_size,
            time_emb_size, class_emb_size, n_classes, n_update,
            message_passing="pna_conv", loader=self.data_loader
        ).to(self.device)
        sampler = CTMCSampler(
            base_dist, LinearDiscreteRateMatrix(
                base_dist, n_atom_tokens, n_bond_tokens).to(self.device),
            n_atom_tokens, n_bond_tokens, TU_MUTAG_CONFIG["max_n_atoms"],
            device=self.device
        )
        model = DeFoG(
            base_dist, noiser, denoiser, sampler, n_atom_tokens, n_bond_tokens,
            TU_MUTAG_CONFIG["max_n_atoms"]
        ).to(self.device)

        lambda_ = x_decoded["lambda_eloss"]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        epochs = self.n_epochs

        for epoch in range(epochs):
            for batch in self.data_loader:
                optimizer.zero_grad()

                loss = model(
                    batch, edge_loss_weight=lambda_,
                    edge_weights=torch.tensor([
                        1.0, x_decoded["dbl_weight"],
                        x_decoded["tpl_weight"], x_decoded["ne_weight"]
                    ])
                )

                loss.backward()
                optimizer.step()

        fc_val_ratio = 0
        for i in range(10):
            with torch.inference_mode():
                model.eval()
                n_samples = 100
                n_steps = 100
                y = torch.cat((
                    torch.ones((n_samples//2,), dtype=torch.long),
                    torch.zeros((n_samples//2,), dtype=torch.long),
                ))

                batch = model.sample(
                    (n_samples,), y=y, n_steps=n_steps,
                    bond_order_bias=(
                        x_decoded["dbl_bias"], x_decoded["tpl_bias"],
                        x_decoded["ne_bias"]
                    ),
                    exit_cap=x_decoded["rate_exit_cap"],
                    temp_scales=(
                        x_decoded["node_scale"], x_decoded["edge_scale"])
                )

            KEK_EDGE_DICT = {
                n: b
                for n, b in TU_MUTAG_CONFIG["edge_dict"].items()
                if n != "aromatic"
            }
            KEK_EDGE_LIST_RDK = list(KEK_EDGE_DICT.values())
            sampled_mols = batch_to_mols(
                batch, TU_MUTAG_CONFIG["node_list"], KEK_EDGE_LIST_RDK,
                TU_MUTAG_CONFIG["max_n_atoms"]
            )

            fc_val = 0
            with rdBase.BlockLogs():
                for mol in sampled_mols:
                    try:
                        mol_val = fix_nitro_charges(mol)
                        if "." not in Chem.MolToSmiles(mol_val):
                            fc_val += 1
                    except:
                        pass
            fc_val_ratio += fc_val / n_samples
        fc_val_ratio /= 10

        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache() 

        return 1. - fc_val_ratio