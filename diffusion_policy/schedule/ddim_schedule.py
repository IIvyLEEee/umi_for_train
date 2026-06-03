import math
from typing import List, Optional, Union

import numpy as np
import torch


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 10,
        num_infer_timesteps: int = 5,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        # beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        set_alpha_to_one: bool = True,
        # clip_sample: bool = True,
        steps_offset: int = 0,
        # prediction_type: str = "epsilon",
        threshold: torch.bfloat16 = 500,
        **kwargs,
    ):
        # config
        self.num_infer_steps = num_infer_timesteps
        self.num_train_steps = num_train_timesteps

        # only use linear for chip testing
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.bfloat16)
        else:
            # self.betas = torch.linspace(
            #     beta_start, beta_end, num_train_timesteps, dtype=torch.bfloat16
            # )
            self.betas = betas_for_alpha_bar(num_train_timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # set timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64)
        )
        self.threshold = threshold

    def step(
        self,
        i,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        variance_noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_infer_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        # print(model_output)
        # print(sample)
        prev_timestep = timestep - self.num_train_steps // self.num_infer_steps
        timestep = timestep + i * (self.num_train_steps // self.num_infer_steps)
        # print(prev_timestep)
        # if i == 1:
        #     print(model_output)
        #     print(sample)
        #     print(prev_timestep)
        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t
        # print(alpha_prod_t)
        # print(alpha_prod_t_prev)
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # only predict epsilon for chip testing
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        # pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # for DDIM: eta = 0
        # variance = self._get_variance(timestep, prev_timestep)
        # std_dev_t = eta * variance ** (0.5)
        std_dev_t = 0.0

        # if use_clipped_model_output:
        #     # the model_output is always re-derived from the clipped x_0 in Glide
        #     model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )
        # if i == 1:
        #     print(prev_sample)
        #     print("  ")

        return prev_sample

    def para_step(
        self,
        batch: int,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        variance_noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # t, t-1, t-2, ..., t-batch+1
        prev_sample = [[] for _ in range(batch)]
        prev_t = []
        for i in range(batch):
            t = timestep - i * self.num_train_steps // self.num_infer_steps
            # print(t)
            # if (t) <= 0:
            #     prev_t.append(torch.full((1,), -40, device="cuda:0"))
            prev_t.append(t - self.num_train_steps // self.num_infer_steps)
            # get previous step value (=t-i-1)
            prev_sample[i] = self.step(
                i,
                model_output,
                t,
                sample,
                eta,
                use_clipped_model_output,
                variance_noise,
            )

        # print(prev_t)
        return prev_sample, prev_t

    def next_timestep(self, refer_samples, pred_samples, prev_t, batch):
        # print(prev_t)
        next_t = prev_t[0]
        t_idx = 0
        for i in range(len(prev_t) - 1):
            err = torch.sum(torch.abs(refer_samples - pred_samples[i + 1]))
            # print("error = ", err)
            if err < self.threshold / math.sqrt(i + 1):
                t_idx = i + 1
                # next_t = prev_t[i + 1] - self.num_train_steps // self.num_infer_steps
                next_t = next_t - self.num_train_steps // self.num_infer_steps
            else:
                break
        print("Succeed", t_idx + 1)
        # print(self.num_infer_steps)
        model_out = pred_samples[t_idx]
        return next_t, model_out, t_idx
        # print(prev_t)
        # return prev_t[1], pred_samples[1], 1

    def set_timesteps(self, num_infer_steps: int, device):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_infer_steps = num_infer_steps
        step_ratio = self.num_train_steps // self.num_infer_steps
        # print(num_infer_steps)
        # print(step_ratio)
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (
            (np.arange(0, num_infer_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps).to(device)
        # self.timesteps += step_ratio
        # print(self.timesteps)


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


if __name__ == "__main__":
    scheduler = DDIMScheduler()
    print(scheduler.betas)
    print(scheduler.alphas)
    print(scheduler.alphas_cumprod)
    # print(scheduler.final_alpha_cumprod)
    # print(scheduler.init_noise_sigma)
    # print(scheduler.num_infer_steps)
    # print(scheduler.timesteps)
