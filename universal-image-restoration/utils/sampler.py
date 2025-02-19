import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Union
import sys
sys.path.insert(0, "/mnt/intern/zhejing.la/code/daclip-uir/universal-image-restoration")
from utils.sde_utils import IRSDE
import cv2
import math

def convert_params(sde, opt):
    def alpha(self):
        thetas_cumsum = torch.cumsum(self.thetas, 0)
        return torch.exp(-thetas_cumsum * self.dt)
    def sigma(self):
        return self.sigma_bars
    def score_fn(self, x, t, text_context=None, image_context=None):
        return self.noise_fn(x, t, text_context=text_context, image_context=image_context)
    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma
    IRSDE.alpha = alpha
    IRSDE.sigma = sigma
    IRSDE.score_fn = score_fn
    IRSDE.prior_sample = noise_state
    irsde = IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=sde.device)
    irsde.N = sde.T
    irsde.sigma_infty = sde.max_sigma
    irsde.set_model(sde.model)
    return irsde


class MRSampler():
    """
    Fast Sampler for Mean Reverting Diffusion
    """
    def __init__(
            self, 
            sde:Union[IRSDE],
            opt,
            num_sample_steps:int,
            solver_type:str="ode",
            parameterization:str="data",
            order:int=2, 
            fit_target:str="data", 
            lambda_min:float=-float('inf'),
            time_schedule:str="linspace",
            timestep_offset:int=0,
            rho:float=7.0,
            thresholding:bool=False,
            threshold_ratio:float=0.995,
            threshold_max:float=1.5,
            denoise_last:bool=False,
            ):
        """
        sde: object of `IRSDE`
        num_sample_steps: int
        solver_type: str, can be ["ode", "sde"]
        parameterization: str, can be ["noise", "data"]
        order: sampler order, can be 1 or 2 or 3 for ode, 1 or 2 for sde
        fit_target: training method, can be "data", "noise", "velocity"
        thresholding: whether to use the "_clip_out" method, only for data prediction
        lambda_min: clipping threshold for the minimum value of `lambda(t)` for numerical stability. This is critical for the cosine (`squaredcos_cap_v2`) noise schedule
        time_schedule: str, can be ["linspace", "leading", "trailing", "karras-n", "karras-t"], according to Table.2 of https://arxiv.org/abs/2305.08891
        timestep_offset: int, offset for the time steps
        rho: float, rho for the karras noise schedule
        threshold_ratio: the ratio for clipping model output, disable if thresholding is False
        threshold_max: max value for clipping model output, disable if thresholding is False
        denoise_last: whether to denoise to zero on the last step
        """
        sde = convert_params(sde, opt)
        self.sde = sde
        assert solver_type in ["ode", "sde"], f"solver_type {solver_type} not supported"
        self.solver_type = solver_type
        assert parameterization in ["noise", "data"], f"network parameterization {parameterization} not supported"
        self.parameterization = parameterization
        if solver_type == "ode":
            assert order in [1, 2, 3], f"order {order} not supported for ode sampler"
        elif solver_type == "sde":
            assert order in [1, 2], f"order {order} not supported for sde sampler"
        self.order = order
        self.lambda_min = lambda_min
        self.fit_target = fit_target
        self.thresholding = thresholding
        self.time_schedule = time_schedule
        self.timestep_offset = timestep_offset
        self.rho = rho
        self.threshold_ratio = threshold_ratio
        self.threshold_max = threshold_max
        self.denoise_last = denoise_last

        # inner parameters
        self.num_training_steps = sde.N
        self.t_array = torch.linspace(0, sde.N, sde.N+1, dtype=torch.float32)
        self.alpha_array = sde.alpha()
        self.sigma_array = sde.sigma()
        self.lambda_array = torch.log(self.alpha_array) - torch.log(self.sigma_array) # 1/2 log SNR
        self._sigmas = self.sigma_array / self.alpha_array # convert to sigma_t in karras' paper
        self.qbuffer = [None]* self.order
        self.set_timesteps(num_sample_steps)
    
    def get_karras_noise_schdule(self, N:int) -> torch.Tensor:
        """
        compute karras noise schedule, descending order
        args:
            N: length of karras sigmas
        return karras sigmas
        """
        # sigma_min = max(0.002, min(self._sigmas).item())
        # sigma_max = min(80, max(self._sigmas).item())
        sigma_min = min(self._sigmas).item()
        sigma_max = max(self._sigmas).item()
        i = torch.linspace(0, 1, N, dtype=self._sigmas.dtype)
        karras_sigmas =(sigma_max**(1/self.rho) + i*(sigma_min**(1/self.rho) - sigma_max**(1/self.rho)))**self.rho
        return karras_sigmas

    def get_karras_time_schedule(self, N:int) -> torch.Tensor:
        t_min = 1.0
        t_max = self.num_training_steps
        i = torch.linspace(0, 1, N, dtype=torch.float32)
        karras_times = (t_max**(1/self.rho) + i*(t_min**(1/self.rho) - t_max**(1/self.rho)))**self.rho
        return karras_times
    
    def sigmas_to_timesteps(self, karras_sigmas:np.ndarray) -> NDArray[int]:
        """
        convert karras sigmas to timesteps, descending order\n
        timesteps[i] = t with karras_sigmas[i]==_sigmas[t]
        """
        _sigmas = self._sigmas.cpu().numpy()
        # get distribution
        dists = karras_sigmas[:,np.newaxis] - _sigmas[np.newaxis,:]

        # get t index
        low_idx = np.cumsum((dists >= 0), axis=1).argmax(axis=1).clip(max=self._sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = _sigmas[low_idx]
        high = _sigmas[high_idx]

        # interpolate sigmas
        w = (low - karras_sigmas) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        timesteps = (1 - w) * low_idx + w * high_idx
        timesteps = list(set(timesteps.round()))
        return np.array(sorted(timesteps, reverse=True)).astype(np.int32)

    def get_linear_lambda(self, N:int) -> torch.Tensor:
        """
        get linear SNR schedule
        """
        snr_min = max(self.lambda_array.min().item(), self.lambda_min)
        snr_max = self.lambda_array.max().item()
        linear_lambda = torch.linspace(snr_min, snr_max, N, dtype=torch.float32)
        timesteps = interpolate_fn(linear_lambda.reshape(-1,1), self.lambda_array.flip(0).reshape(1,-1).cpu(), self.t_array.flip(0).reshape(1,-1)).reshape(-1)
        return timesteps
    
    def set_timesteps(self, num_inference_steps:int):
        """
        set the time steps for sampling, run before sampling
        num_inference_steps: int, number of inference time steps
        """
        # clip minimum of lambda_array for numerical stability
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_array, [0]), self.lambda_min)
        max_timestep = (self.num_training_steps - clipped_idx).item()
        # init time steps
        if self.time_schedule == "linspace":
            timesteps = torch.linspace(0, max_timestep, num_inference_steps+1, dtype=torch.float32).flip(0)
        elif self.time_schedule == "leading":
            step_interval = max_timestep // (num_inference_steps+1)
            timesteps = (torch.arange(0, num_inference_steps+1, dtype=torch.float32)* step_interval).flip(0)
            timesteps += self.timestep_offset
        elif self.time_schedule == "trailing":
            step_interval = self.num_training_steps / (num_inference_steps+1)
            timesteps = torch.arange(max_timestep, 0, -step_interval, dtype=torch.float32)
        elif self.time_schedule == "karras-n":
            karras_sigmas = self.get_karras_noise_schdule(num_inference_steps+1)
            # timesteps = self.sigmas_to_timesteps(karras_sigmas)
            timesteps = interpolate_fn(karras_sigmas.reshape(-1,1), self._sigmas.reshape(1,-1).cpu(), self.t_array.reshape(1,-1)).reshape(-1)
        elif self.time_schedule == "karras-t":
            timesteps = self.get_karras_time_schedule(num_inference_steps+1)
        elif self.time_schedule == "linear-lambda":
            timesteps = self.get_linear_lambda(num_inference_steps+1)
        else:
            raise ValueError(f"time_schedule {self.time_schedule} not supported")
        
        # get interpolation of alpha, sigma, lambda on timesteps
        # sigmas[i] <-- sigmas[timesteps[i]]
        self.alphas = interpolate_fn(timesteps.reshape(-1,1), self.t_array.reshape(1,-1), self.alpha_array.reshape(1,-1).cpu()).reshape(-1)
        self.sigmas = interpolate_fn(timesteps.reshape(-1,1), self.t_array.reshape(1,-1), self.sigma_array.reshape(1,-1).cpu()).reshape(-1)
        self.lambdas = interpolate_fn(timesteps.reshape(-1,1), self.t_array.reshape(1,-1), self.lambda_array.reshape(1,-1).cpu()).reshape(-1)
        self.timesteps = timesteps.tolist()

    def update_qbuffer(self, new_output):
        """update qbuffer after one output step"""
        for i in reversed(range(self.order)):
            if i == 0:
                self.qbuffer[0] = new_output
            else:
                self.qbuffer[i] = self.qbuffer[i-1]

    def _clip_output(self, output:torch.Tensor):
        """
        Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing pixels from saturation at each step. We find that dynamic thresholding results in significantly better photorealism as well as better image-text alignment, especially when using very large guidance weights.
        s = threshold_ratio * max(abs(output))
        according to https://arxiv.org/abs/2205.11487
        """
        dtype = output.dtype
        b, c, *remaining_dims = output.shape

        if dtype not in (torch.float32, torch.float64):
            output = output.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten output for doing quantile calculation along each image
        output = output.reshape(b, c * np.prod(remaining_dims))
        s = torch.quantile(output.abs(), self.threshold_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.threshold_max)
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        output = torch.clamp(output, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        output = output.reshape(b, c, *remaining_dims)
        output = output.to(dtype)
        return output

    def convert_model_output(self, score, xti, i, cond):
        """according to the fit_target, convert model output to noise prediction or data prediction"""
        if self.parameterization == "data":
            # convert model output to data prediction
            if self.fit_target == "noise" or self.fit_target == "reverse_traj":
                output = (xti - self.sigmas[i]* score)/self.alphas[i]
                if isinstance(self.sde, IRSDE):
                    output -= (1 - self.alphas[i])/self.alphas[i] * cond
            elif self.fit_target == "data" or self.fit_target == "patch":
                output = score
            elif self.fit_target == "velocity":
                if isinstance(self.sde, IRSDE):
                    output = self.alphas[i] * xti - self.sigmas[i]/self.sde.sigma_infty * score + (1 - self.alphas[i]) * cond
                else:
                    output = self.alphas[i] * xti - self.sigmas[i] * score
            else:
                raise ValueError(f"{self.fit_target} should be noise, data or velocity")
        else: # parameterization == "noise"
            # convert model output to noise prediction
            if self.fit_target == "noise" or self.fit_target == "reverse_traj":
                output = score
            elif self.fit_target == "data":
                output = (xti - cond - (score - cond)*self.alphas[i])/self.sigmas[i]
            elif self.fit_target == "velocity":
                output = (score * self.alphas[i] + (xti - cond) * self.sigmas[i]/self.sde.sigma_infty) / self.sde.sigma_infty
            else:
                raise ValueError(f"{self.fit_target} should be noise, data or velocity")
        
        if self.thresholding:
            output = self._clip_output(output)
        return output

        
    def step_ode_3rd(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor) -> torch.Tensor:
        """execute one step of MRSampler-ode-3 ++
        args:
            xti_1: torch.Tensor, x(t_{i-1})
            qbuffer: list, [model(i-1), model(i-2), model(i-3)]
            i: int, index of timestep
            mu: only useful when mrsde
        """
        score = self.sde.score_fn(xti_1, self.timesteps[i-1], text_context=self.text_context, image_context=self.image_context)
        self.update_qbuffer(self.convert_model_output(score, xti_1, i-1, mu))

        hi = self.lambdas[i] - self.lambdas[i-1]
        hi_1 = self.lambdas[i-1] - self.lambdas[i-2]
        hi_2 = self.lambdas[i-2] - self.lambdas[i-3]
        ri = hi_1 / hi
        ri_1 = hi_2 / hi
        D1_0 = (self.qbuffer[0] - self.qbuffer[1]) / ri
        D1_1 = (self.qbuffer[1] - self.qbuffer[2]) / ri_1
        D1 = D1_0 + (D1_0 - D1_1) * ri / (ri + ri_1)
        D2 = (D1_0 - D1_1) / (ri + ri_1)
        if self.parameterization == "data":
            phi1 = torch.exp(-hi)-1
            phi2 = phi1 / hi + 1
            phi3 = phi2 / hi - 0.5
            D = phi1 * self.qbuffer[0] - phi2 * D1 + phi3 * D2
            xti = self.sigmas[i]/self.sigmas[i-1] * xti_1 - self.alphas[i] * D
            xti += mu * (1 - self.sigmas[i]/self.sigmas[i-1] 
                            + self.sigmas[i]/self.sigmas[i-1]*self.alphas[i-1]
                            - self.alphas[i])
        else: # parameterization == "noise"
            phi1 = torch.exp(hi)-1
            phi2 = phi1 / hi - 1
            phi3 = phi2 / hi - 0.5
            D = phi1 * self.qbuffer[0] + phi2 * D1 + phi3 * D2
            xti = self.alphas[i]/self.alphas[i-1] * xti_1 - self.sigmas[i] * D
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1])
        return xti

    def step_ode_2nd(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor) -> torch.Tensor:
        """
        execute one step of MRSampler-ode-2
        args:
            xti_1: torch.Tensor, x(t_{i-1})
            qbuffer: list, [model(i-1), model(i-2)]
            i: int, index of timestep
            mu: only useful when mrsde
        return:
            xti: torch.Tensor, x(t_i)
        """
        score = self.sde.score_fn(xti_1, self.timesteps[i-1], text_context=self.text_context, image_context=self.image_context)
        self.update_qbuffer(self.convert_model_output(score, xti_1, i-1, mu))

        hi = self.lambdas[i] - self.lambdas[i-1]
        hi_1 = self.lambdas[i-1] - self.lambdas[i-2]
        r = hi_1 / hi
        D1 = (self.qbuffer[0] - self.qbuffer[1])/r
        if self.parameterization == "data":
            phi1 = torch.exp(-hi)-1
            phi2 = phi1 / hi + 1
            D = phi1 * self.qbuffer[0] - phi2 * D1
            xti = self.sigmas[i]/self.sigmas[i-1] * xti_1 - self.alphas[i] * D
            xti += mu * (1 - self.sigmas[i]/self.sigmas[i-1] 
                            + self.sigmas[i]/self.sigmas[i-1]*self.alphas[i-1]
                            - self.alphas[i])
        else: # parameterization == "noise"
            phi1 = torch.exp(hi)-1
            phi2 = phi1 / hi - 1
            D = phi1 * self.qbuffer[0] + phi2 * D1
            xti = self.alphas[i]/self.alphas[i-1] * xti_1 - self.sigmas[i] * D
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1])
        return xti
    

    def step_ode_1st(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor) -> torch.Tensor:
        """
        execute one step of MRSampler-ode-1
        args:
            xti_1: torch.Tensor, x(t_{i-1})
            qbuffer: list, [model(i-1), model(i-2)]
            i: int, index of timestep
            mu: conditional images
        return:
            xti: torch.Tensor, x(t_i)
        """
        score = self.sde.score_fn(xti_1, self.timesteps[i-1], text_context=self.text_context, image_context=self.image_context)
        self.update_qbuffer(self.convert_model_output(score, xti_1, i-1, mu))
        hi = self.lambdas[i] - self.lambdas[i-1]

        if self.parameterization == "data":
            xti = self.sigmas[i]/self.sigmas[i-1] * xti_1 - self.alphas[i] * (torch.exp(-hi)-1) * self.qbuffer[0]
            xti += mu * (1 - self.sigmas[i]/self.sigmas[i-1] 
                            + self.sigmas[i]/self.sigmas[i-1]*self.alphas[i-1]
                            - self.alphas[i])
        else: # parameterization == "noise"
            xti = self.alphas[i]/self.alphas[i-1] * xti_1 - self.sigmas[i] * (torch.exp(hi)-1) * self.qbuffer[0]
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1])
        return xti
    
    def step_sde_1st(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor) -> torch.Tensor:
        """
        execute one step of MRSampler-sde-1
        args:
            xti_1: torch.Tensor, x(t_{i-1})
            qbuffer: list, [model(i-1), model(i-2)]
            i: int, index of timestep
            mu: conditional images
        return:
            xti: torch.Tensor, x(t_i)
        """
        score = self.sde.score_fn(xti_1, self.timesteps[i-1], text_context=self.text_context, image_context=self.image_context)
        self.update_qbuffer(self.convert_model_output(score, xti_1, i-1, mu))
        hi = self.lambdas[i] - self.lambdas[i-1]
        noise = torch.randn_like(xti_1)

        if self.parameterization == "data":
            phi1 = 1 - torch.exp(-2*hi)
            xti = self.sigmas[i]/self.sigmas[i-1]*torch.exp(-hi) * xti_1 + self.alphas[i] * phi1 * self.qbuffer[0] + self.sigmas[i] * torch.sqrt(phi1) * noise
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1]*torch.exp(-2*hi) - self.alphas[i] * phi1)
        else: # parameterization == "noise"
            xti = self.alphas[i]/self.alphas[i-1] * xti_1 - 2 * self.sigmas[i] * (torch.exp(hi)-1) * self.qbuffer[0] + self.sigmas[i] * torch.sqrt(torch.exp(2*hi)-1) * noise
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1])
        return xti
    
    def step_sde_2nd(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor) -> torch.Tensor:
        """
        execute one step of MRSampler-sde-2
        args:
            xti_1: torch.Tensor, x(t_{i-1})
            qbuffer: list, [model(i-1), model(i-2)]
            i: int, index of timestep
            mu: conditional images
        return:
            xti: torch.Tensor, x(t_i)
        """
        score = self.sde.score_fn(xti_1, self.timesteps[i-1], text_context=self.text_context, image_context=self.image_context)
        self.update_qbuffer(self.convert_model_output(score, xti_1, i-1, mu))
        hi = self.lambdas[i] - self.lambdas[i-1]
        hi_1 = self.lambdas[i-1] - self.lambdas[i-2]
        r = hi_1 / hi
        D1 = (self.qbuffer[0] - self.qbuffer[1])/r
        noise = torch.randn_like(xti_1)

        if self.parameterization == "data":
            phi1 = 1 - torch.exp(-2*hi)
            phi2 = 1 - phi1/2/hi
            xti = self.sigmas[i]/self.sigmas[i-1]*torch.exp(-hi) * xti_1 + self.alphas[i] * phi1 * self.qbuffer[0] + self.alphas[i] * phi2 * D1 + self.sigmas[i] * torch.sqrt(phi1) * noise
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1]*torch.exp(-2*hi) - self.alphas[i] * phi1)
        else: # parameterization == "noise"
            phi1 = torch.exp(hi)-1
            phi2 = phi1 / hi - 1
            xti = self.alphas[i]/self.alphas[i-1] * xti_1 - 2 * self.sigmas[i] * phi1 * self.qbuffer[0] - 2* self.sigmas[i] * phi2 * D1 + self.sigmas[i] * torch.sqrt(torch.exp(2*hi)-1) * noise
            xti += mu * (1 - self.alphas[i]/self.alphas[i-1])
        return xti
    
    def step(self, xti_1:torch.Tensor, i:int, mu:torch.Tensor, order:int) -> torch.Tensor:
        if self.solver_type == "ode":
            if order == 1:
                return self.step_ode_1st(xti_1, i, mu)
            elif order == 2:
                return self.step_ode_2nd(xti_1, i, mu)
            elif order == 3:
                return self.step_ode_3rd(xti_1, i, mu)
            else:
                raise ValueError(f"order {order} not supported")
        elif self.solver_type == "sde":
            if order == 1:
                return self.step_sde_1st(xti_1, i, mu)
            elif order == 2:
                return self.step_sde_2nd(xti_1, i, mu)
            else :
                raise ValueError(f"order {order} not supported")
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
    
    def sampling(self, cond:torch.Tensor, text_context=None, image_context=None, use_tqdm=False, position=0, savedir=None, noisy_state=None):
        """
        execute sampling
        args:
            cond: condition images
            use_tqdm: whether to use tqdm
            position: position of the progress bar
        return:
            samples: torch.Tensor, [-1,1]
            qbuffer: list, [model(0), model(1), ...]
            states: list, [..., x(t_M-1), x(t_M)]
        """
        self.sde.set_mu(cond)
        self.text_context = text_context
        self.image_context = image_context
        def update_refresh(pbar):
            if pbar is not None: 
                pbar.update(1)
                pbar.refresh()
        def save_img(img, path):
            img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)
        assert hasattr(self, "timesteps"), f"timesteps not set, call `set_timesteps` before sampling"
        pbar = tqdm(desc="sampling", total=len(self.timesteps)-1, position=position, colour="green", leave=False) if use_tqdm else None
        M = len(self.timesteps)-1
        qbuffer = []
        states = []
        with torch.no_grad():
            # prior sampling
            t0 = self.timesteps[0]
            xti = self.sde.prior_sample(cond) if noisy_state is None else noisy_state
            if savedir is not None:
                save_img(xti, f"{savedir}/prior_sample.png")
                states.append(xti.cpu())
            # first few steps use lower-order sampling
            for i in range(1, self.order):
                xti = self.step(xti, i, cond, order=i)
                update_refresh(pbar)
                if savedir is not None:
                    save_img(xti, f"{savedir}/step_{i}.png")
                    qbuffer.append(self.qbuffer[0].cpu())
                    states.append(xti.cpu())
            # step from order to M-1
            for i in range(self.order, M):
                if M < 10:
                    order = min(self.order, M-i)
                else:
                    order = self.order
                xti = self.step(xti, i, cond, order=order)
                if torch.isnan(xti).any():
                    print("nan detected", i)
                    break
                update_refresh(pbar)
                if savedir is not None:
                    save_img(xti, f"{savedir}/step_{i}.png")
                    qbuffer.append(self.qbuffer[0].cpu())
                    states.append(xti.cpu())
            if self.denoise_last:
                # last step denoise to zero
                score = self.sde.score_fn(xti, self.timesteps[M-1], text_context=self.text_context, image_context=self.image_context)
                x0 = self.convert_model_output(score, xti, M-1, cond)
            else:
                x0 = self.step(xti, M, cond, order=self.order)
            update_refresh(pbar)
            if savedir is not None:
                save_img(x0, f"{savedir}/step_{M}.png")
                qbuffer.append(self.qbuffer[0].cpu())
                states.append(x0.cpu())
        # post processing
        if pbar is not None:
            pbar.close()
        samples = torch.clip(x0, 0, 1.0)
        return samples, qbuffer, states




# copied from Cheng Lu's DPM-Solver (https://github.com/LuChengTHU/dpm-solver/blob/main/dpm_solver_pytorch.py#L1253)
def interpolate_fn(x:torch.Tensor, xp:torch.Tensor, yp:torch.Tensor):
    """
    A piecewise linear interpolation function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints, need to be monotonically increasing.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

class HeunSampler():
    def __init__(self, sde:IRSDE, nfe:int, solver_type:str="sde"):
        self.sde = sde
        self.nfe = nfe
        self.solver_type = solver_type
        self.dt = sde.dt
        self.timesteps = np.linspace(sde.T, 0, nfe+1, dtype=float).tolist()
        t_array = torch.linspace(0, sde.T, sde.T+1, dtype=torch.float32)
        self.thetas = interpolate_fn(torch.tensor(self.timesteps).reshape(-1,1), t_array.reshape(1,-1), sde.thetas.cpu().reshape(1,-1)).reshape(-1)
        self.sigmas = interpolate_fn(torch.tensor(self.timesteps).reshape(-1,1), t_array.reshape(1,-1), sde.sigmas.cpu().reshape(1,-1)).reshape(-1)
    
    def score_fn(self, x, i, text_context=None, image_context=None):
        epsilon = self.sde.noise_fn(x, self.timesteps[i], text_context=self.text_context, image_context=self.image_context)
        return -epsilon / self.sigmas[i]

    def step1st(self, mu, xs, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t), score(s)
        """
        score = self.score_fn(xs, s, text_context=self.text_context, image_context=self.image_context)
        if self.solver_type == "sde":
            f = self.thetas[s] * (mu - xs) - self.sigmas[s]**2 * score
            g = self.sigmas[s]
            dw = math.sqrt((t-s)*self.dt) * torch.randn_like(xs)
            xt = xs - f * (t-s)*self.dt + g * dw
        elif self.solver_type == "ode":
            f = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * score
            xt = xs - f * (t-s)*self.dt
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return xt, score
    
    def step2nd(self, mu, xs, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t)
        """
        xhat, score0 = self.step1st(mu, xs, s, t)
        score1 = self.score_fn(xhat, t, text_context=self.text_context, image_context=self.image_context)
        if self.solver_type == "sde":
            f0 = self.thetas[s] * (mu - xs) - self.sigmas[s]**2 * score0
            f1 = self.thetas[t] * (mu - xhat) - self.sigmas[t]**2 * score1
            g = 0.5* (self.sigmas[s]+self.sigmas[t])
            dw = math.sqrt((t-s)*self.dt) * torch.randn_like(xs)
            xt = xs - 0.5 * (f0 + f1) * (t-s)*self.dt + g * dw
        elif self.solver_type == "ode":
            f0 = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * score0
            f1 = self.thetas[t] * (mu - xhat) - 0.5 * self.sigmas[t]**2 * score1
            xt = xs - 0.5 * (f0 + f1) * (t-s)*self.dt
        else:
            raise ValueError(f"solver_type {self.solver_type} not supported")
        return xt
    
    def sampling(self, cond, text_context=None, image_context=None, noisy_state=None):
        self.sde.set_mu(cond)
        self.text_context = text_context
        self.image_context = image_context
        states = []
        M = len(self.timesteps)-1 # M=NFE
        pbar = tqdm(desc="Heun", total=M, position=0, colour="blue", leave=False)
        with torch.no_grad():
            xti = self.sde.noise_state(cond) if noisy_state is None else noisy_state
            states.append(xti.cpu())
            for i in range(2, M+1, 2):
                xti = self.step2nd(cond, xti, i-2, i)
                pbar.update(2)
                states.append(xti.cpu())
            if M % 2 == 1:
                xti,_ = self.step1st(cond, xti, M-1, M)
                pbar.update(1)
                states.append(xti.cpu())
        pbar.close()
        x0 = torch.clip(xti, 0, 1.0)
        return x0, states

class AdamsMoulton():
    def __init__(self, sde:IRSDE, order, nfe:int):
        self.sde = sde
        self.nfe = nfe
        self.order = order-1
        self.dt = sde.dt
        self.timesteps = np.linspace(sde.T, 0, nfe+1, dtype=float).tolist()
        t_array = torch.linspace(0, sde.T, sde.T+1, dtype=torch.float32)
        self.thetas = interpolate_fn(torch.tensor(self.timesteps).reshape(-1,1), t_array.reshape(1,-1), sde.thetas.cpu().reshape(1,-1)).reshape(-1)
        self.sigmas = interpolate_fn(torch.tensor(self.timesteps).reshape(-1,1), t_array.reshape(1,-1), sde.sigmas.cpu().reshape(1,-1)).reshape(-1)
        self.qbuffer = [None]*self.order
        self.xbuffer = [None]*self.order
    
    def update_buffer(self, new_output, x):
        """update buffer after one output step"""
        for i in reversed(range(self.order)):
            if i == 0:
                self.qbuffer[0] = new_output
                self.xbuffer[0] = x
            else:
                self.qbuffer[i] = self.qbuffer[i-1]
                self.xbuffer[i] = self.xbuffer[i-1]

    def score_fn(self, x, i, text_context=None, image_context=None):
        epsilon = self.sde.noise_fn(x, self.timesteps[i], text_context=self.text_context, image_context=self.image_context)
        return -epsilon / self.sigmas[i]

    def step1st(self, mu, xs, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t), score(s)
        """
        score = self.score_fn(xs, s, text_context=self.text_context, image_context=self.image_context)
        self.update_buffer(score, xs)
        f = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * score
        xt = xs - f * (t-s)*self.dt
        return xt
    
    def step2nd(self, mu, xs, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t)
        """
        xhat = self.step1st(mu, xs, s, t)
        score1 = self.score_fn(xhat, t, text_context=self.text_context, image_context=self.image_context)
        f0 = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * self.qbuffer[0]
        f1 = self.thetas[t] * (mu - xhat) - 0.5 * self.sigmas[t]**2 * score1
        xt = xs - 0.5 * (f0 + f1) * (t-s)*self.dt
        return xt
    
    def step3rd(self, mu, xs, s_1, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t)
        """
        xhat = self.step1st(mu, xs, s, t)
        score1 = self.score_fn(xhat, t, text_context=self.text_context, image_context=self.image_context)
        
        f0 = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * self.qbuffer[0]
        f1 = self.thetas[t] * (mu - xhat) - 0.5 * self.sigmas[t]**2 * score1
        f_1 = self.thetas[s_1] * (mu - self.xbuffer[1]) - 0.5 * self.sigmas[s_1]**2 * self.qbuffer[1]
        xt = xs - (5*f1 + 8*f0 - f_1) * (t-s)*self.dt/12
        return xt
    
    def step4th(self, mu, xs, s_2, s_1, s, t):
        """
        args:
            mu: torch.Tensor, conditional images
            xs: torch.Tensor, x(start time)
            s: int, index of start time
            t: int, index of end time
        return:
            x(t)
        """
        xhat = self.step1st(mu, xs, s, t)
        score1 = self.score_fn(xhat, t, text_context=self.text_context, image_context=self.image_context)
        
        f0 = self.thetas[s] * (mu - xs) - 0.5 * self.sigmas[s]**2 * self.qbuffer[0]
        f1 = self.thetas[t] * (mu - xhat) - 0.5 * self.sigmas[t]**2 * score1
        f_1 = self.thetas[s_1] * (mu - self.xbuffer[1]) - 0.5 * self.sigmas[s_1]**2 * self.qbuffer[1]
        f_2 = self.thetas[s_2] * (mu - self.xbuffer[2]) - 0.5 * self.sigmas[s_2]**2 * self.qbuffer[2]
        xt = xs - (9*f1 + 19*f0 - 5*f_1 + f_2) * (t-s)*self.dt/24
        return xt
    
    def sampling(self, cond, text_context=None, image_context=None, noisy_state=None):
        self.sde.set_mu(cond)
        self.text_context = text_context
        self.image_context = image_context
        states = []
        M = len(self.timesteps)-1 # M=NFE
        pbar = tqdm(desc="Adams-Moulton", total=M, position=0, colour="blue", leave=False)
        with torch.no_grad():
            xti = self.sde.noise_state(cond) if noisy_state is None else noisy_state
            states.append(xti.cpu())
            for i in range(0, M-1, 2):
                if i>=4:
                    xti = self.step4th(cond, xti, i-4, i-2, i, i+2)
                elif i>=2:
                    xti = self.step3rd(cond, xti, i-2, i, i+2)
                else:
                    xti = self.step2nd(cond, xti, i, i+2)
                pbar.update(2)
                states.append(xti.cpu())
            if M % 2 == 1:
                xti = self.step1st(cond, xti, M-1, M)
                pbar.update(1)
                states.append(xti.cpu())
        pbar.close()
        x0 = torch.clip(xti, 0, 1.0)
        return x0, states