import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import parse as V

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel as BaseESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def get_similarity_weight(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list."""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            idx_i = int(utt_i[-1]) - 1
            spk_i = utt_i.split("_")[idx_i].split("-")[0]

            idx_j = int(utt_j[-1]) - 1
            spk_j = utt_j.split("_")[idx_j].split("-")[0]

            weight[i, j] = int(spk_i == spk_j)

    return weight


def get_similarity_weight_wsj2mix(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list. (for wsj2mix data)"""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            spk_i = utt_i.split("_")[-1][:3]
            spk_j = utt_j.split("_")[-1][:3]

            weight[i, j] = int(spk_i == spk_j)

    return weight


def get_similarity_weight_ami(utt_list: List[str]):
    """Get speaker similarity weight based on the utt_list. (for ami data)"""
    weight = torch.zeros(len(utt_list), len(utt_list))
    for i, utt_i in enumerate(utt_list):
        for j, utt_j in enumerate(utt_list):
            spk_i = utt_i.split("_")[3]
            spk_j = utt_j.split("_")[3]

            weight[i, j] = int(spk_i == spk_j)

    return weight


def get_speaker_labels(utt_list: List[str], is_wsj2mix: bool = False, is_ami: bool = False):
    """Extract speaker labels from utterance IDs for AAM-Softmax."""
    speaker_labels = []
    speaker_to_id = {}
    current_id = 0
    
    for utt in utt_list:
        if is_wsj2mix:
            spk = utt.split("_")[-1][:3]
        elif is_ami:
            spk = utt.split("_")[3]
        else:
            idx = int(utt[-1]) - 1
            spk = utt.split("_")[idx].split("-")[0]
        
        if spk not in speaker_to_id:
            speaker_to_id[spk] = current_id
            current_id += 1
        
        speaker_labels.append(speaker_to_id[spk])
    
    return torch.tensor(speaker_labels, dtype=torch.long)


class TgtSpkQformerESPnetASRModel_V2(BaseESPnetASRModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == enroll.shape[0]
            == enroll_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encode(
            speech, speech_lengths, enroll, enroll_lengths
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        stats = dict()

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            prompt_lens = spk_prompt.size(1)
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out[:, prompt_lens:],
                encoder_out_lens - prompt_lens,
                text,
                text_lengths,
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2b. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            enroll: (Batch, Dim) for embedding enrollment
                    or (Batch, Length) for speech enrollment
            enroll_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Extract feats for enrollment
            enroll_feats, enroll_feats_lengths = self._extract_feats(
                enroll, enroll_lengths
            )

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 5. Forward encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encoder(
            feats, feats_lengths, enroll_feats, enroll_feats_lengths
        )

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        return encoder_out, encoder_out_lens, spk_prompt, enroll_embedding

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        spk_prompt: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, spk_prompt
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_aam_softmax_loss(
        self,
        enroll_emb: torch.Tensor,
        speaker_labels: torch.Tensor,
    ):
        """Calculate AAM-Softmax loss for speaker classification with ASP pooling"""
        
        # Initialize ASP pooling if not done yet
        if self.asp_pooling is None:
            emb_dim = enroll_emb.size(-1)
            # Use warm-up gamma value and enable projection
            current_gamma = self.get_current_asp_gamma()
            self.asp_pooling = AttentiveStatisticsPooling(
                input_dim=emb_dim,
                gamma=current_gamma,
                use_projection=True  # Enable projection to match mean pooling dimension
            )
            self.asp_pooling = self.asp_pooling.to(enroll_emb.device)
        else:
            # Update gamma for existing ASP pooling
            current_gamma = self.get_current_asp_gamma()
            self.asp_pooling.gamma = current_gamma
        
        # Apply ASP pooling: (batch_size, seq_len, dim) -> (batch_size, dim) with normalization
        pooled_emb = self.asp_pooling(enroll_emb)
        
        # Initialize classifier if not done yet
        if self.aam_classifier is None:
            pooled_dim = pooled_emb.size(-1)  # dim (same as input due to projection)
            self.aam_classifier = torch.nn.Linear(pooled_dim, self.num_speakers, bias=False)
            self.aam_classifier = self.aam_classifier.to(pooled_emb.device)
        
        # Normalize features and weights
        normalized_features = F.normalize(pooled_emb.float(), dim=-1)
        normalized_weights = F.normalize(self.aam_classifier.weight, dim=-1)
        
        # Calculate cosine similarity
        cosine_sim = F.linear(normalized_features, normalized_weights)
        
        # Calculate current margin with warm-up
        if self.current_epoch < self.warm_up_epochs:
            current_margin = 0.0
        else:
            current_margin = self.aam_margin
        
        # Apply additive angular margin
        cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine_sim)
        
        # Create one-hot target mask
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, speaker_labels.view(-1, 1), 1.0)
        
        # Add margin only to target class
        theta += one_hot * current_margin
        cosine_sim_with_margin = torch.cos(theta)
        
        # Scale with temperature
        logits = cosine_sim_with_margin / self.aam_temp
        logits = logits.type_as(pooled_emb)
        
        # Calculate cross-entropy loss
        loss_aam = F.cross_entropy(logits, speaker_labels)
        
        # Calculate accuracy
        pred_aam = logits.argmax(dim=-1)
        acc_aam = float(torch.sum(pred_aam == speaker_labels)) / float(speaker_labels.size(0))
        
        return loss_aam, acc_aam


class TgtSpkQformerESPnetASRModel_V4(TgtSpkQformerESPnetASRModel_V2):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        contrastive_type: str = "w2v2",
        contrastive_weight: float = 1.0,
        contrastive_temp: float = 0.1,
        num_negatives: int = 10,
        is_wsj2mix: bool = False,
        is_ami: bool = False,
        # AAM-Softmax parameters
        num_speakers: int = 1000,  # Number of speakers in the dataset
        aam_softmax_weight: float = 0.4,  # Will be multiplied by contrastive_weight
        aam_margin: float = 0.25,  # Final margin value
        aam_temp: float = 0.0333,  # Temperature for AAM-Softmax (s=30)
        warm_up_epochs: int = 5,  # Number of epochs for margin warm-up
        # ASP pooling parameters
        asp_attention_dim: int = 128,  # Hidden dimension for ASP attention
        asp_gamma: float = 6.0,  # Temperature parameter for ASP attention (final value)
        asp_gamma_warmup_epochs: int = 6,  # Number of epochs for ASP gamma warm-up
        asp_gamma_initial: float = 1.0,  # Initial gamma value for warm-up
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            aux_ctc=aux_ctc,
            ctc_weight=ctc_weight,
            interctc_weight=interctc_weight,
            ignore_id=ignore_id,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
            report_cer=report_cer,
            report_wer=report_wer,
            sym_space=sym_space,
            sym_blank=sym_blank,
            sym_sos=sym_sos,
            sym_eos=sym_eos,
            extract_feats_in_collect_stats=extract_feats_in_collect_stats,
            lang_token_id=lang_token_id,
        )

        self.contrastive_type = contrastive_type
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.num_negatives = num_negatives
        self.is_wsj2mix = is_wsj2mix
        self.is_ami = is_ami
        
        # AAM-Softmax parameters
        self.num_speakers = num_speakers
        self.aam_softmax_weight = aam_softmax_weight
        self.aam_margin = aam_margin
        self.aam_temp = aam_temp
        self.warm_up_epochs = warm_up_epochs
        self.current_epoch = 0  # Track current epoch for margin warm-up
        
        # AAM-Softmax head - assumes enrollment embedding dimension from encoder
        # We'll get the dimension from the first forward pass
        self.aam_classifier = None
        
        # ASP pooling layer - will be initialized on first forward pass
        self.asp_pooling = None
        self.asp_attention_dim = asp_attention_dim  # Hidden dimension for attention network
        self.asp_gamma = asp_gamma  # Temperature parameter for attention (final value)
        self.asp_gamma_warmup_epochs = asp_gamma_warmup_epochs  # Warm-up epochs for gamma
        self.asp_gamma_initial = asp_gamma_initial  # Initial gamma value

        logging.info(f"Speaker prompt for encoder: {self.encoder.use_spk_prompt}")
        logging.info(f"Speaker prompt for decoder: {self.decoder.use_spk_prompt}")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        enroll: torch.Tensor,
        enroll_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            enroll: (Batch, Length, ...)
            enroll_lengths: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == enroll.shape[0]
            == enroll_lengths.shape[0]
        ), (
            speech.shape,
            speech_lengths.shape,
            text.shape,
            text_lengths.shape,
            enroll.shape,
            enroll_lengths.shape,
        )

        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # compute negtive sample probability, (batch, batch)
        if self.is_wsj2mix:
            sim_weight = get_similarity_weight_wsj2mix(kwargs["utt_id"])
        elif self.is_ami:
            sim_weight = get_similarity_weight_ami(kwargs["utt_id"])
        else:
            sim_weight = get_similarity_weight(kwargs["utt_id"])
        neg_weight = torch.ones_like(sim_weight).masked_fill_(sim_weight == 1, -10000)
        neg_weight = F.softmax(neg_weight, dim=1)

        # 1. Encoder
        encoder_out, encoder_out_lens, spk_prompt, enroll_embedding = self.encode(
            speech, speech_lengths, enroll, enroll_lengths
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_con, acc_con = None, None
        loss_aam, acc_aam = None, None
        stats = dict()

        # Extract speaker labels for AAM-Softmax
        speaker_labels = get_speaker_labels(kwargs["utt_id"], self.is_wsj2mix, self.is_ami)
        speaker_labels = speaker_labels.to(enroll_embedding.device)

        # 2a. contrastive Loss
        if self.contrastive_weight > 0.0:
            if self.contrastive_type == "w2v2":
                loss_con, acc_con = self._calc_w2v2_contrastive_loss(
                    spk_prompt, enroll_embedding, neg_weight
                )
            else:
                raise NotImplementedError(f"contrastive_type={self.contrastive_type}")

            # collect contrastive branch stats
            stats["loss_con"] = loss_con.detach() if loss_con is not None else None
            stats["acc_con"] = acc_con if acc_con is not None else None

        # 2a2. AAM-Softmax Loss
        if self.contrastive_weight > 0.0 and self.aam_softmax_weight > 0.0:
            loss_aam, acc_aam = self._calc_aam_softmax_loss(
                enroll_embedding, speaker_labels
            )
            
            # collect AAM-Softmax branch stats
            stats["loss_aam"] = loss_aam.detach() if loss_aam is not None else None
            stats["acc_aam"] = acc_aam if acc_aam is not None else None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            prompt_encoder = self.encoder.use_spk_prompt
            prompt_lens = spk_prompt.size(1)
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out[:, prompt_lens:] if prompt_encoder else encoder_out,
                encoder_out_lens - prompt_lens if prompt_encoder else encoder_out_lens,
                text,
                text_lengths,
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 2c. Attention decoder branch
        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, spk_prompt
        )

        # 3. CTC-Att loss definition
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        if self.contrastive_weight > 0.0:
            loss = loss + self.contrastive_weight * loss_con
            
        # Add AAM-Softmax loss with weight 0.4 * contrastive_weight
        if self.contrastive_weight > 0.0 and self.aam_softmax_weight > 0.0:
            aam_weight = self.aam_softmax_weight * self.contrastive_weight
            loss = loss + aam_weight * loss_aam

        # Collect Attn branch stats
        stats["loss_att"] = loss_att.detach() if loss_att is not None else None
        stats["acc"] = acc_att
        stats["cer"] = cer_att
        stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_w2v2_contrastive_loss(
        self,
        spk_prompt: torch.Tensor,
        enroll_emb: torch.Tensor,
        neg_weight: torch.Tensor,
    ):
        """Calculate Arc-InfoNCE loss for contrastive learning with ASP pooling"""
        
        # Initialize ASP pooling if not done yet
        if self.asp_pooling is None:
            emb_dim = enroll_emb.size(-1)
            # Use warm-up gamma value and enable projection
            current_gamma = self.get_current_asp_gamma()
            self.asp_pooling = AttentiveStatisticsPooling(
                input_dim=emb_dim,
                gamma=current_gamma,
                use_projection=True  # Enable projection to match mean pooling dimension
            )
            self.asp_pooling = self.asp_pooling.to(enroll_emb.device)
        else:
            # Update gamma for existing ASP pooling
            current_gamma = self.get_current_asp_gamma()
            self.asp_pooling.gamma = current_gamma
        
        # Apply ASP pooling to enrollment embeddings: (batch_size, seq_len, dim) -> (batch_size, dim) with normalization
        pooled_enroll_emb = self.asp_pooling(enroll_emb)
        
        # For speaker prompt, use simple mean pooling with normalization
        pooled_spk_prompt = spk_prompt.mean(dim=1)  # (batch_size, dim)
        pooled_spk_prompt = F.normalize(pooled_spk_prompt, dim=-1)  # Add L2 normalization for consistency

        # get negative enrollment embeddings
        batch_size = pooled_spk_prompt.size(0)
        neg_enroll_emb = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(
                neg_weight[b], self.num_negatives, replacement=True
            )
            neg_enroll_emb.append(pooled_enroll_emb[neg_idx])
        # (num_negatives, batch_size, dim)
        neg_enroll_emb = torch.stack(neg_enroll_emb, dim=1)

        # (1 + num_negatives, batch_size, dim)
        target_enroll_emb = torch.cat([pooled_enroll_emb.unsqueeze(0), neg_enroll_emb], dim=0)

        # Both speaker prompt and enrollment embeddings now have same dimension (dim)
        # Both are already normalized (ASP has built-in normalization, speaker prompt is manually normalized)
        spk_prompt_norm = pooled_spk_prompt  # Already normalized
        target_enroll_emb_norm = target_enroll_emb  # ASP output is already normalized

        # compute cosine similarity, (1 + num_negatives, batch_size)
        cosine_sim = torch.cosine_similarity(
            spk_prompt_norm, target_enroll_emb_norm, dim=-1
        )
        
        # Apply angular margin for Arc-InfoNCE
        cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine_sim)
        # Add margin only to positive sample (index 0)
        theta[0] = theta[0] + 0.15  # margin = 0.15
        cosine_sim = torch.cos(theta)
        
        # Scale with temperature and reshape
        logits = cosine_sim / self.contrastive_temp
        logits = logits.type_as(pooled_spk_prompt)
        # (batch_size, 1 + num_negatives)
        logits = logits.transpose(0, 1).contiguous()
        
        # generate targets, (batch_size) - target is always index 0 (positive sample)
        target = logits.new_zeros(logits.size(0), dtype=torch.long)

        # compute Arc-InfoNCE loss
        loss_con = F.cross_entropy(logits, target)
        # compute contrastive accuracy
        pred_con = logits.argmax(dim=-1)
        acc_con = float(torch.sum(pred_con == target)) / float(target.size(0))

        return loss_con, acc_con

    def set_epoch(self, epoch: int):
        """Set current epoch for AAM-Softmax margin warm-up"""
        self.current_epoch = epoch

    def get_current_asp_gamma(self):
        """Calculate current ASP gamma value with warm-up"""
        if self.current_epoch < self.asp_gamma_warmup_epochs:
            # Linear warm-up from initial to final gamma
            progress = self.current_epoch / self.asp_gamma_warmup_epochs
            current_gamma = self.asp_gamma_initial + progress * (self.asp_gamma - self.asp_gamma_initial)
        else:
            current_gamma = self.asp_gamma
        return current_gamma


class AttentiveStatisticsPooling(torch.nn.Module):
    """Attentive Statistics Pooling (ASP) layer
    
    Computes attention weights and performs statistics pooling as:
    p̃ = normalize(1/L * Σ_i p̃_i)
    s_t = p̃^T ê_t
    α_t = softmax(γ * s_t) where γ ≈ 5~10
    
    Args:
        input_dim: Input feature dimension
        gamma: Temperature parameter for attention (default: 5.0)
        use_projection: Whether to use projection layer to reduce dimension back to input_dim (default: True)
    """
    
    def __init__(self, input_dim: int, gamma: float = 5.0, use_projection: bool = True, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.gamma = gamma
        self.use_projection = use_projection
        
        # Projection layer to reduce concatenated features back to input dimension
        if self.use_projection:
            self.projection = torch.nn.Linear(input_dim * 2, input_dim)
            # Initialize projection layer
            torch.nn.init.xavier_uniform_(self.projection.weight)
            torch.nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for ASP following the exact formula from the image
        
        Args:
            x: Input features (batch_size, seq_len, feat_dim)
            lengths: Sequence lengths (batch_size,), optional
            
        Returns:
            pooled_features: Statistics pooled features
                           - If use_projection=True: (batch_size, feat_dim) with L2 normalization
                           - If use_projection=False: (batch_size, feat_dim * 2) without normalization
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Step 1: Compute mean vector p̃ = normalize(1/L * Σ_i ê_i)
        if lengths is not None:
            # Use actual sequence lengths for mean computation
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            
            # Masked mean: 1/L * Σ_i ê_i
            p_tilde = (x * mask).sum(dim=1) / lengths.unsqueeze(1).float()  # (batch_size, feat_dim)
        else:
            # Simple mean over all frames: 1/L * Σ_i ê_i
            p_tilde = x.mean(dim=1)  # (batch_size, feat_dim)
        
        # Normalize the mean vector: p̃ = normalize(1/L * Σ_i ê_i)
        p_tilde = F.normalize(p_tilde, dim=-1)  # (batch_size, feat_dim)
        
        # Step 2: Compute attention scores s_t = p̃^T ê_t
        # Expand p_tilde for broadcasting: (batch_size, 1, feat_dim)
        p_tilde_expanded = p_tilde.unsqueeze(1)
        
        # Compute dot product attention scores: s_t = p̃^T ê_t
        attention_scores = torch.sum(p_tilde_expanded * x, dim=-1)  # (batch_size, seq_len)
        
        # Step 3: Apply temperature and softmax: α_t = exp(γ*s_t) / Σ_k exp(γ*s_k)
        attention_scores = self.gamma * attention_scores
        
        # Apply length mask if available
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        
        # Step 4: Compute weighted mean (first moment): μ = Σ_t α_t ê_t
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_mean = torch.sum(attention_weights_expanded * x, dim=1)  # (batch_size, feat_dim)
        
        # Step 5: Compute weighted standard deviation (second moment)
        # Following the formula: σ = √(max(m₂ - μ ⊙ μ, 0) + ε)
        # where m₂ = Σ_t α_t(ê_t ⊙ ê_t) is the second moment
        
        # Compute second moment: m₂ = Σ_t α_t(ê_t ⊙ ê_t)
        second_moment = torch.sum(attention_weights_expanded * (x * x), dim=1)  # (batch_size, feat_dim)
        
        # Compute variance: m₂ - μ ⊙ μ
        variance = second_moment - weighted_mean * weighted_mean  # (batch_size, feat_dim)
        
        # Apply max(variance, 0) + ε for numerical stability
        variance = torch.clamp(variance, min=0.0) + 1e-8
        
        # Compute standard deviation: σ = √(variance)
        weighted_std = torch.sqrt(variance)  # (batch_size, feat_dim)
        
        # Step 6: Concatenate mean and standard deviation
        pooled_features = torch.cat([weighted_mean, weighted_std], dim=-1)  # (batch_size, feat_dim * 2)
        
        # Step 7: Optional projection and normalization
        if self.use_projection:
            # Project back to original dimension: z_e^ASP = norm([μ; σ]) ∈ ℝ^2d
            # Linear mapping W ∈ ℝ^d×2d projects z_e = norm(W[μ; σ])
            projected_features = self.projection(pooled_features)  # (batch_size, feat_dim)
            # Apply L2 normalization: z_e = norm(W[μ; σ])
            pooled_features = F.normalize(projected_features, dim=-1)  # (batch_size, feat_dim)
        
        return pooled_features
