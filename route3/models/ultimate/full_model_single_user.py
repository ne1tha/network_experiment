from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from channels import ChannelState, DifferentiableChannelSimulator
from losses import ConditionalSSDDDecoderLoss, DecoderLossOutput

from .channel_modnet import ChannelModNet
from .code_mask import CodeMaskModule, CodeMaskOutput
from .decoder_refine_gan import EnhancementLossOutput, EnhancementStatus, PerceptualAdversarialEnhancer
from .decoder_ssdd import ConditionalSSDDDecoder, DecoderOutput
from .dual_path_encoder import UltimateDualPathEncoder
from .fusion_interface import DualPathEncoderOutput
from .rate_modnet import RateModNet
from .semantic_distill import DistillationOutput, SemanticDistillationModule


@dataclass(frozen=True)
class BranchTransmission:
    branch_name: str
    latent: torch.Tensor
    modulated: torch.Tensor
    channel_gates: torch.Tensor
    rate_scores: torch.Tensor
    tx: torch.Tensor
    rx: torch.Tensor
    mask: CodeMaskOutput
    channel_state: ChannelState


@dataclass(frozen=True)
class SingleUserTransmissionOutput:
    encoder: DualPathEncoderOutput
    semantic: BranchTransmission
    detail: BranchTransmission
    distillation: DistillationOutput | None = None
    base_reconstruction: DecoderOutput | None = None
    final_reconstruction: DecoderOutput | None = None
    reconstruction: DecoderOutput | None = None
    base_decoder_losses: DecoderLossOutput | None = None
    final_decoder_losses: DecoderLossOutput | None = None
    decoder_losses: DecoderLossOutput | None = None
    enhancement_losses: EnhancementLossOutput | None = None
    enhancement_status: EnhancementStatus | None = None


class AdaptiveTransmissionBlock(nn.Module):
    def __init__(self, channels: int, channel_type: str = "awgn", enforce_pruning: bool = True):
        super().__init__()
        self.channel_modnet = ChannelModNet(channels=channels)
        self.rate_modnet = RateModNet(channels=channels)
        self.code_mask = CodeMaskModule(enforce_pruning=enforce_pruning)
        self.channel = DifferentiableChannelSimulator(channel_type=channel_type)

    def forward(self, latent: torch.Tensor, snr_db: torch.Tensor | float, rate_ratio: torch.Tensor | float, branch_name: str) -> BranchTransmission:
        modulated, channel_gates = self.channel_modnet(latent, snr_db=snr_db)
        rate_scores = self.rate_modnet(modulated, rate_ratio=rate_ratio)
        mask_output = self.code_mask(modulated, channel_scores=rate_scores, rate_ratio=rate_ratio)
        tx = mask_output.masked
        rx, channel_state = self.channel(tx, snr_db=snr_db)

        if torch.any(mask_output.rate_ratio < 1.0) and torch.allclose(tx, modulated):
            raise RuntimeError(f"{branch_name} rate masking did not change the transmitted symbols.")

        return BranchTransmission(
            branch_name=branch_name,
            latent=latent,
            modulated=modulated,
            channel_gates=channel_gates,
            rate_scores=rate_scores,
            tx=tx,
            rx=rx,
            mask=mask_output,
            channel_state=channel_state,
        )


class UltimateSingleUserTransmission(nn.Module):
    """Route-3 phase-2 single-user encoder plus adaptive transmission path."""

    def __init__(
        self,
        encoder: UltimateDualPathEncoder | None = None,
        semantic_channel_type: str = "awgn",
        detail_channel_type: str = "awgn",
        semantic_distillation: SemanticDistillationModule | None = None,
        decoder: ConditionalSSDDDecoder | None = None,
        decoder_loss: ConditionalSSDDDecoderLoss | None = None,
        enhancer: PerceptualAdversarialEnhancer | None = None,
    ):
        super().__init__()
        self.encoder = encoder or UltimateDualPathEncoder()
        self.semantic_tx = AdaptiveTransmissionBlock(channels=192, channel_type=semantic_channel_type)
        self.detail_tx = AdaptiveTransmissionBlock(channels=128, channel_type=detail_channel_type)
        self.semantic_distillation = semantic_distillation
        self.decoder = decoder
        self.decoder_loss = decoder_loss
        self.enhancer = enhancer

    def forward(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor | float,
        sem_rate_ratio: torch.Tensor | float,
        det_rate_ratio: torch.Tensor | float,
        decode_stochastic: bool = True,
        decode_timestep: torch.Tensor | float = 1.0,
        noise_image: torch.Tensor | None = None,
        run_enhancement: bool = True,
        compute_enhancement_discriminator_loss: bool = False,
        enhancement_enable_perceptual: bool | None = None,
        enhancement_enable_adversarial: bool | None = None,
    ) -> SingleUserTransmissionOutput:
        encoder_output = self.encoder(x)
        semantic = self.semantic_tx(encoder_output.z_sem, snr_db=snr_db, rate_ratio=sem_rate_ratio, branch_name="semantic")
        detail = self.detail_tx(encoder_output.z_det, snr_db=snr_db, rate_ratio=det_rate_ratio, branch_name="detail")
        distillation = None
        if self.semantic_distillation is not None:
            distillation = self.semantic_distillation(x, encoder_output.sem_pyramid)
        base_reconstruction = None
        final_reconstruction = None
        reconstruction = None
        base_decoder_losses = None
        final_decoder_losses = None
        decoder_losses = None
        enhancement_losses = None
        enhancement_status = (
            self.enhancer.status(
                enable_perceptual=enhancement_enable_perceptual,
                enable_adversarial=enhancement_enable_adversarial,
            )
            if self.enhancer is not None
            else None
        )
        if self.decoder is not None:
            base_reconstruction = self.decoder(
                rx_sem=semantic.rx,
                rx_det=detail.rx,
                sem_state=semantic.channel_state,
                det_state=detail.channel_state,
                sem_rate_ratio=sem_rate_ratio,
                det_rate_ratio=det_rate_ratio,
                output_size=encoder_output.input_size,
                noise_image=noise_image,
                decode_timestep=decode_timestep,
                stochastic=decode_stochastic,
            )
            final_reconstruction = base_reconstruction
            reconstruction = final_reconstruction
            if self.decoder_loss is not None:
                base_decoder_losses = self.decoder_loss(
                    x_gt=x,
                    x_hat=base_reconstruction.x_hat,
                    noisy_input=base_reconstruction.noisy_input,
                    predicted_residual=base_reconstruction.predicted_residual,
                )
                final_decoder_losses = base_decoder_losses
                decoder_losses = final_decoder_losses
            if self.enhancer is not None and run_enhancement:
                enhancement_losses = self.enhancer.generator_step(
                    x_gt=x,
                    base_reconstruction=base_reconstruction,
                    compute_discriminator_loss=compute_enhancement_discriminator_loss,
                    enable_perceptual=enhancement_enable_perceptual,
                    enable_adversarial=enhancement_enable_adversarial,
                )
                final_reconstruction = enhancement_losses.refined_reconstruction
                reconstruction = final_reconstruction
                enhancement_status = enhancement_losses.status
                if self.decoder_loss is not None:
                    # The refiner does not predict a fresh decoder residual, so final-image supervision
                    # only applies image-domain structural losses to the refined output.
                    final_decoder_losses = self.decoder_loss(
                        x_gt=x,
                        x_hat=final_reconstruction.x_hat,
                    )
                    decoder_losses = final_decoder_losses
        elif self.enhancer is not None:
            raise RuntimeError("Perceptual or adversarial enhancement requires the main decoder to be enabled.")
        return SingleUserTransmissionOutput(
            encoder=encoder_output,
            semantic=semantic,
            detail=detail,
            distillation=distillation,
            base_reconstruction=base_reconstruction,
            final_reconstruction=final_reconstruction,
            reconstruction=reconstruction,
            base_decoder_losses=base_decoder_losses,
            final_decoder_losses=final_decoder_losses,
            decoder_losses=decoder_losses,
            enhancement_losses=enhancement_losses,
            enhancement_status=enhancement_status,
        )
