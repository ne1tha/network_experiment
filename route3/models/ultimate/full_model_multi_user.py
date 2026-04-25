from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from losses import DecoderLossOutput

from .decoder_refine_gan import EnhancementLossOutput, EnhancementStatus
from .decoder_ssdd import DecoderOutput
from .full_model_single_user import SingleUserTransmissionOutput, UltimateSingleUserTransmission
from .multiuser_bandwidth import BandwidthAllocationOutput, SemanticBandwidthAllocator
from .multiuser_pairing import PairingOutput, SemanticPairingAllocator
from .sfma_cua import CUAOutput, CrossUserAttention


@dataclass(frozen=True)
class MultiUserOutput:
    single_user: SingleUserTransmissionOutput
    pairing: PairingOutput
    semantic_cua: CUAOutput
    bandwidth: BandwidthAllocationOutput
    shared_rx_sem: torch.Tensor
    private_rx_det: torch.Tensor
    base_shared_reconstruction: DecoderOutput | None = None
    final_shared_reconstruction: DecoderOutput | None = None
    shared_reconstruction: DecoderOutput | None = None
    shared_decoder_losses: DecoderLossOutput | None = None
    shared_enhancement_losses: EnhancementLossOutput | None = None
    shared_enhancement_status: EnhancementStatus | None = None


class UltimateMultiUserTransmission(nn.Module):
    """Route-3 phase-7 multi-user semantic sharing stack. Only z_sem is shared."""

    def __init__(
        self,
        single_user_model: UltimateSingleUserTransmission | None = None,
        semantic_cua: CrossUserAttention | None = None,
        pairing_allocator: SemanticPairingAllocator | None = None,
        bandwidth_allocator: SemanticBandwidthAllocator | None = None,
    ):
        super().__init__()
        self.single_user_model = single_user_model or UltimateSingleUserTransmission()
        self.semantic_cua = semantic_cua or CrossUserAttention(channels=192, num_heads=4)
        self.pairing_allocator = pairing_allocator or SemanticPairingAllocator()
        self.bandwidth_allocator = bandwidth_allocator or SemanticBandwidthAllocator()

    def forward(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        semantic_bandwidth_budget: float,
        decode_stochastic: bool = True,
        decode_timestep: torch.Tensor | float = 1.0,
        noise_image: torch.Tensor | None = None,
        run_enhancement: bool = True,
        compute_enhancement_discriminator_loss: bool = False,
        enhancement_enable_perceptual: bool | None = None,
        enhancement_enable_adversarial: bool | None = None,
    ) -> MultiUserOutput:
        single_user = self.single_user_model(
            x,
            snr_db=snr_db,
            sem_rate_ratio=sem_rate_ratio,
            det_rate_ratio=det_rate_ratio,
            decode_stochastic=decode_stochastic,
            decode_timestep=decode_timestep,
            noise_image=noise_image,
            run_enhancement=False,
            compute_enhancement_discriminator_loss=False,
            enhancement_enable_perceptual=enhancement_enable_perceptual,
            enhancement_enable_adversarial=enhancement_enable_adversarial,
        )

        z_sem = single_user.encoder.z_sem
        z_det = single_user.encoder.z_det
        if z_sem.shape[0] % 2 != 0:
            raise ValueError(f"Multi-user semantic sharing expects an even number of users, got {z_sem.shape[0]}.")

        semantic_vectors = z_sem.flatten(2).mean(dim=2)
        pairing = self.pairing_allocator(semantic_vectors)
        semantic_cua = self.semantic_cua(z_sem, pairing.pair_indices)
        bandwidth = self.bandwidth_allocator(pairing.pair_costs, total_budget=semantic_bandwidth_budget)
        per_user_bandwidth = bandwidth.per_user_bandwidth.to(device=z_sem.device, dtype=z_sem.dtype)
        mean_bandwidth = per_user_bandwidth.mean().clamp_min(1e-8)
        relative_scale = (per_user_bandwidth / mean_bandwidth).view(-1, 1, 1, 1)
        absolute_gate = (per_user_bandwidth / (per_user_bandwidth + 1.0)).view(-1, 1, 1, 1)
        bandwidth_shared_semantic = semantic_cua.shared_semantic * relative_scale * absolute_gate

        if torch.equal(bandwidth_shared_semantic, z_det):
            raise RuntimeError("Detected an invalid route where z_det entered semantic sharing.")

        if not torch.equal(single_user.detail.rx, single_user.detail.rx.clone()):
            raise RuntimeError("Detail features changed unexpectedly during private-path validation.")

        base_shared_reconstruction = None
        final_shared_reconstruction = None
        shared_reconstruction = None
        shared_decoder_losses = None
        shared_enhancement_losses = None
        shared_enhancement_status = (
            self.single_user_model.enhancer.status(
                enable_perceptual=enhancement_enable_perceptual,
                enable_adversarial=enhancement_enable_adversarial,
            )
            if self.single_user_model.enhancer is not None
            else None
        )

        if self.single_user_model.decoder is not None:
            shared_noise_image = noise_image
            if shared_noise_image is None and single_user.base_reconstruction is not None:
                shared_noise_image = single_user.base_reconstruction.noisy_input
            base_shared_reconstruction = self.single_user_model.decoder(
                rx_sem=bandwidth_shared_semantic,
                rx_det=single_user.detail.rx,
                sem_state=single_user.semantic.channel_state,
                det_state=single_user.detail.channel_state,
                sem_rate_ratio=sem_rate_ratio,
                det_rate_ratio=det_rate_ratio,
                output_size=single_user.encoder.input_size,
                noise_image=shared_noise_image,
                decode_timestep=decode_timestep,
                stochastic=decode_stochastic,
            )
            final_shared_reconstruction = base_shared_reconstruction
            shared_reconstruction = final_shared_reconstruction
            if self.single_user_model.decoder_loss is not None:
                shared_decoder_losses = self.single_user_model.decoder_loss(
                    x_gt=x,
                    x_hat=base_shared_reconstruction.x_hat,
                    noisy_input=base_shared_reconstruction.noisy_input,
                    predicted_residual=base_shared_reconstruction.predicted_residual,
                )
            if self.single_user_model.enhancer is not None and run_enhancement:
                shared_enhancement_losses = self.single_user_model.enhancer.generator_step(
                    x_gt=x,
                    base_reconstruction=base_shared_reconstruction,
                    compute_discriminator_loss=compute_enhancement_discriminator_loss,
                    enable_perceptual=enhancement_enable_perceptual,
                    enable_adversarial=enhancement_enable_adversarial,
                )
                final_shared_reconstruction = shared_enhancement_losses.refined_reconstruction
                shared_reconstruction = final_shared_reconstruction
                shared_enhancement_status = shared_enhancement_losses.status

        return MultiUserOutput(
            single_user=single_user,
            pairing=pairing,
            semantic_cua=semantic_cua,
            bandwidth=bandwidth,
            shared_rx_sem=bandwidth_shared_semantic,
            private_rx_det=single_user.detail.rx,
            base_shared_reconstruction=base_shared_reconstruction,
            final_shared_reconstruction=final_shared_reconstruction,
            shared_reconstruction=shared_reconstruction,
            shared_decoder_losses=shared_decoder_losses,
            shared_enhancement_losses=shared_enhancement_losses,
            shared_enhancement_status=shared_enhancement_status,
        )
