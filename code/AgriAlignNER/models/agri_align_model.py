import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet152
import math

class EntityLevelDynamicGatedAlignment(nn.Module):

    def __init__(
        self,
        text_dim,
        visual_dim,
        hidden_dim=768
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.entity_proj = nn.Linear(
            text_dim,
            hidden_dim
        )

        self.visual_proj = nn.Linear(
            visual_dim,
            hidden_dim
        )

        self.gate_fc = nn.Linear(
            hidden_dim + 1,
            1
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(
            self,
            entity_features,
            visual_features,
            entity_valid_mask=None
    ):

        """
        entity_features:
            (batch, num_entities, hidden)

        visual_features:
            (batch, num_regions, hidden)
        """

        batch_size = entity_features.size(0)


        # Entity query
        Q = self.entity_proj(
            entity_features
        )

        # Visual key
        K = self.visual_proj(
            visual_features
        )

        similarity = torch.matmul(
            Q,
            K.transpose(1,2)
        )

        similarity = (
            similarity /
            math.sqrt(self.hidden_dim)
        )

        if entity_valid_mask is not None:
            similarity = similarity.masked_fill(
                entity_valid_mask.unsqueeze(-1) == 0,
                -1e4
            )

        attention_weights = F.softmax(
            similarity,
            dim=-1
        )

        visual_context = torch.matmul(
            attention_weights,
            visual_features
        )


        max_similarity,_ = similarity.max(
            dim=-1,
            keepdim=True
        )


        gate_input = torch.cat(
            [
                Q,
                max_similarity
            ],
            dim=-1
        )


        gate_weights = torch.sigmoid(
            self.gate_fc(
                gate_input
            )
        )

        enhanced_entities = self.norm(
            entity_features
            +
            gate_weights *
            visual_context
        )


        return (
            enhanced_entities,
            gate_weights,
            attention_weights
        )

class AdaptiveMultiGranularityAlignment(nn.Module):

    def __init__(
            self,
            text_dim,
            visual_dim,
            hidden_dim=768,
            num_heads=8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.token_level_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )


        self.entity_level_edga = EntityLevelDynamicGatedAlignment(
            text_dim=text_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim
        )


        # projection layers

        self.text_proj = nn.Linear(
            text_dim,
            hidden_dim
        )

        self.visual_proj = nn.Linear(
            visual_dim,
            hidden_dim
        )


        # =====================================================
        # Adaptive fusion gate
        # token aligned + entity aligned token feature
        # =====================================================

        self.granularity_gate = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            ),

            nn.Sigmoid()
        )


        # =====================================================
        # Contrastive learning projection
        # =====================================================

        self.contrast_proj = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                128
            )
        )


        self.output_proj = nn.Linear(
            hidden_dim,
            text_dim
        )


        self.norm = nn.LayerNorm(
            text_dim
        )

    def forward(
            self,
            token_features,
            entity_features,
            entity_spans,
            visual_features,
            entity_mask=None,
            entity_valid_mask=None
    ):


        """
        Args:

            token_features:
                BERT token representation

                (batch, seq_len, hidden)


            entity_features:
                aggregated entity representations

                (batch, num_entities, hidden)


            entity_spans:
                entity token positions

                [
                    [(start,end),...],
                    ...
                ]


            visual_features:
                global/grid/ROI visual regions

                (batch,num_regions,hidden)

        """


        batch_size, seq_len, _ = token_features.shape


        token_hidden = self.text_proj(
            token_features
        )


        visual_hidden = self.visual_proj(
            visual_features
        )


        token_aligned, token_attention = self.token_level_attn(
            query=token_hidden,
            key=visual_hidden,
            value=visual_hidden
        )

        entity_aligned, \
        gate_weights, \
        entity_attention = self.entity_level_edga(
            entity_features,
            visual_features,
            entity_valid_mask
        )


        entity_token_features = self.entity_to_token(
            entity_aligned,
            entity_spans,
            seq_len
        )


        fusion_input = torch.cat(
            [
                token_aligned,
                entity_token_features
            ],
            dim=-1
        )


        alpha = self.granularity_gate(
            fusion_input
        )


        fused_features = (
                alpha * token_aligned
                +
                (1-alpha)
                *
                entity_token_features
        )



        output = self.output_proj(
            fused_features
        )


        output = self.norm(
            token_features + output
        )

        contrastive_loss = self._compute_contrastive_loss(
            token_hidden,
            visual_hidden,
            entity_mask
        )


        return (
            output,
            contrastive_loss,
            alpha,
            entity_attention
        )

    def entity_to_token(
            self,
            entity_features,
            entity_spans,
            seq_len
    ):


        batch_size = entity_features.size(0)


        token_entity_features = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_dim,
            device=entity_features.device
        )


        for b in range(batch_size):

            spans = entity_spans[b]


            for idx, (start,end) in enumerate(spans):

                token_entity_features[
                    b,
                    start:end
                ] = entity_features[
                    b,
                    idx
                ]


        return token_entity_features

    def _compute_contrastive_loss(
            self,
            text_features,
            visual_features,
            entity_mask=None
    ):


        if entity_mask is not None:


            mask = entity_mask.unsqueeze(-1).float()


            mask_sum = mask.sum(
                dim=1
            ).clamp(
                min=1e-8
            )


            text_global = (
                text_features * mask
            ).sum(dim=1) / mask_sum


        else:

            text_global = text_features.mean(
                dim=1
            )



        visual_global = visual_features.mean(
            dim=1
        )



        text_proj = self.contrast_proj(
            text_global
        )


        visual_proj = self.contrast_proj(
            visual_global
        )



        text_proj = F.normalize(
            text_proj,
            dim=-1
        )


        visual_proj = F.normalize(
            visual_proj,
            dim=-1
        )



        logits = torch.matmul(
            text_proj,
            visual_proj.transpose(0,1)
        ) / 0.07



        labels = torch.arange(
            logits.size(0),
            device=logits.device
        )



        loss = F.cross_entropy(
            logits,
            labels
        )


        return loss

class BoundaryDetectionModule(nn.Module):
    """
    Dynamic auxiliary boundary detection module.

    The number of output labels is determined by
    boundary_label_mapping:

        BMO   -> 3 labels
        BIO   -> 9 labels
        BIOES -> 17 labels
    """

    def __init__(
        self,
        hidden_dim,
        boundary_label_mapping,
    ):
        super().__init__()

        if not isinstance(
            boundary_label_mapping,
            dict,
        ):
            raise TypeError(
                "boundary_label_mapping must be a dict."
            )

        if "O" not in boundary_label_mapping:
            raise ValueError(
                "boundary_label_mapping must contain "
                "the outside label 'O'."
            )

        self.boundary_label_mapping = dict(
            boundary_label_mapping
        )

        self.num_boundary_labels = len(
            self.boundary_label_mapping
        )

        # 检查标签ID是否从0连续编号
        boundary_ids = sorted(
            self.boundary_label_mapping.values()
        )

        expected_ids = list(
            range(self.num_boundary_labels)
        )

        if boundary_ids != expected_ids:
            raise ValueError(
                "Boundary label IDs must be consecutive "
                f"from 0 to "
                f"{self.num_boundary_labels - 1}, "
                f"but got {boundary_ids}."
            )

        self.boundary_fc = nn.Linear(
            hidden_dim,
            self.num_boundary_labels,
        )

        self.boundary_crf = CRF(
            self.num_boundary_labels,
            batch_first=True,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        boundary_labels=None,
    ):
        """
        Args:
            hidden_states:
                (batch_size, seq_len, hidden_dim)

            attention_mask:
                (batch_size, seq_len)

            boundary_labels:
                (batch_size, seq_len)

        Returns:
            boundary_tags:
                CRF-decoded auxiliary label sequences.

            boundary_loss:
                Auxiliary CRF loss.
        """
        emissions = self.boundary_fc(
            hidden_states
        )

        crf_mask = attention_mask.bool()

        boundary_loss = None

        if boundary_labels is not None:
            valid_labels = boundary_labels[
                crf_mask
            ]

            if valid_labels.numel() > 0:
                min_label_id = int(
                    valid_labels.min().item()
                )
                max_label_id = int(
                    valid_labels.max().item()
                )

                if min_label_id < 0:
                    raise RuntimeError(
                        f"Negative boundary label ID: "
                        f"{min_label_id}."
                    )

                if (
                    max_label_id
                    >= self.num_boundary_labels
                ):
                    raise RuntimeError(
                        f"Boundary label ID "
                        f"{max_label_id} exceeds "
                        f"configured label number "
                        f"{self.num_boundary_labels}."
                    )

            boundary_loss = -self.boundary_crf(
                emissions,
                boundary_labels,
                mask=crf_mask,
                reduction="mean",
            )

        boundary_tags = self.boundary_crf.decode(
            emissions,
            mask=crf_mask,
        )

        return boundary_tags, boundary_loss

class EntityAggregator(nn.Module):

    def __init__(self):
        super().__init__()


    def extract_spans(self, tags):
        """
        Extract entity spans from B-M-O boundary tags.

        Args:
            tags:
                predicted boundary tags
                e.g. [0,1,2,2,0,1,2,0]

        Returns:
            spans:
                [(start,end), ...]
        """

        spans = []

        start = None


        for i, tag in enumerate(tags):

            # B: begin of entity
            if tag == 1:

                # close previous entity
                if start is not None:
                    spans.append(
                        (start, i)
                    )

                start = i


            # M: middle of entity
            elif tag == 2:

                continue


            # O: outside
            else:

                if start is not None:

                    spans.append(
                        (start, i)
                    )

                    start = None



        # close last entity
        if start is not None:

            spans.append(
                (start, len(tags))
            )


        return spans



    def forward(
        self,
        token_features,
        boundary_tags,
        attention_mask
    ):

        """
        Aggregate token representations into entity representations.

        Args:

            token_features:
                BERT output

                (batch, seq_len, hidden)


            boundary_tags:
                CRF decoded boundary labels

                list:
                [
                  [0,1,2,0,...],
                  [...]
                ]


        Returns:

            entity_features:

                (batch,max_entities,hidden)


            entity_spans:

                entity token positions


            entity_mask:

                valid entity indicator
        """

        batch_entities = []
        batch_spans = []
        batch_has_entity = []
        max_entities = 0



        # ==============================
        # 1. Extract entity representations
        # ==============================

        for b in range(
            token_features.size(0)
        ):


            spans = self.extract_spans(
                boundary_tags[b]
            )


            entities = []


            for start,end in spans:


                entity_vector = token_features[
                    b,
                    start:end
                ].mean(
                    dim=0
                )


                entities.append(
                    entity_vector
                )


            # No entity case
            has_entity = True

            if len(entities) == 0:
                has_entity = False

                entities.append(
                    torch.zeros(
                        token_features.size(-1),
                        device=token_features.device
                    )
                )

                spans = [
                    (0, 0)
                ]

            entities = torch.stack(
                entities
            )

            batch_entities.append(
                entities
            )

            batch_spans.append(
                spans
            )

            batch_has_entity.append(
                has_entity
            )

            max_entities = max(
                max_entities,
                entities.size(0)
            )

        # ==============================
        # 2. Padding entities
        # ==============================

        padded_entities=[]

        entity_masks=[]



        for idx, entities in enumerate(batch_entities):


            entity_num = entities.size(0)


            pad_num = (
                max_entities
                -
                entity_num
            )


            if pad_num > 0:


                padding=torch.zeros(
                    pad_num,
                    entities.size(-1),
                    device=entities.device
                )


                entities=torch.cat(
                    [
                        entities,
                        padding
                    ],
                    dim=0
                )



            padded_entities.append(
                entities
            )


            # valid entity mask
            mask=torch.zeros(
                max_entities,
                device=entities.device
            )

            if batch_has_entity[idx]:
                mask[:entity_num] = 1

            entity_masks.append(
                mask
            )



        # ==============================
        # 3. Stack batch
        # ==============================

        entity_features=torch.stack(
            padded_entities
        )


        entity_mask=torch.stack(
            entity_masks
        )



        return (
            entity_features,
            batch_spans,
            entity_mask
        )

class AgriAlignNERModel(nn.Module):


    def __init__(
            self,
            label_list,
            args,
            boundary_label_mapping,
            use_amgca=True,
    ):
        super().__init__()
        self.args = args
        self.use_amgca = use_amgca

        self.num_labels = len(label_list)

        self.boundary_label_mapping = dict(
            boundary_label_mapping
        )

        self.num_boundary_labels = len(
            self.boundary_label_mapping
        )

        if "O" not in self.boundary_label_mapping:
            raise ValueError(
                "boundary_label_mapping must "
                "contain label 'O'."
            )

        self.boundary_o_id = (
            self.boundary_label_mapping["O"]
        )

        self.boundary_id_to_label = {
            label_id: label_name
            for label_name, label_id
            in self.boundary_label_mapping.items()
        }

        # Text Encoder
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        hidden_dim = self.bert_config.hidden_size  # 768

        # Visual Encoder
        if args.use_prompt:
            self.visual_encoder = RegionLevelVisualEncoder(output_dim=hidden_dim)

            self.boundary_detector = (
                BoundaryDetectionModule(
                    hidden_dim=hidden_dim,
                    boundary_label_mapping=(
                        self.boundary_label_mapping
                    ),
                )
            )

            self.entity_aggregator = EntityAggregator()

            # Cross-Modal Fusion
            if use_amgca:
                self.cross_modal_fusion = AdaptiveMultiGranularityAlignment(
                    text_dim=hidden_dim,
                    visual_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
            else:
                self.cross_modal_fusion = EntityLevelDynamicGatedAlignment(
                    text_dim=hidden_dim,
                    visual_dim=hidden_dim,
                    hidden_dim=hidden_dim
                )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.lambda_boundary = getattr(
            args,
            "boundary_weight",
            0.3,
        )

        self.lambda_contrastive = getattr(
            args,
            "contrastive_weight",
            0.05,
        )
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, images=None, aux_imgs=None, boundary_labels=None):

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        text_features = bert_output['last_hidden_state']  # (batch, seq_len, hidden)
        contrastive_loss = 0
        boundary_loss = 0
        if self.args.use_prompt and images is not None:

            boundary_tags, b_loss = self.boundary_detector(
                text_features, attention_mask, boundary_labels
            )
            if b_loss is not None:
                boundary_loss = b_loss

            entity_features, entity_spans, entity_valid_mask = self.entity_aggregator(
                text_features,
                boundary_tags,
                attention_mask
            )
            visual_features = self.visual_encoder(images, aux_imgs)
            if self.use_amgca:
                if self.training and boundary_labels is not None:
                    tags_for_mask = boundary_labels
                else:
                    tags_for_mask = boundary_tags
                entity_mask = self._create_entity_mask(tags_for_mask, attention_mask)
                fused_features, c_loss, alpha, attn = self.cross_modal_fusion(
                    token_features=text_features,
                    entity_features=entity_features,
                    entity_spans=entity_spans,
                    visual_features=visual_features,
                    entity_mask=entity_mask,
                    entity_valid_mask=entity_valid_mask
                )
                contrastive_loss = c_loss
            else:
                fused_features, gate_weights, attn = self.cross_modal_fusion(
                    entity_features,
                    visual_features,
                    entity_valid_mask
                )
        else:
            fused_features = text_features


        fused_features = self.dropout(fused_features)
        lstm_output, _ = self.bilstm(fused_features)
        emissions = self.fc(lstm_output)
        ner_mask = attention_mask.bool()
        logits = self.crf.decode(
            emissions,
            mask=ner_mask,
        )
        loss = None
        if boundary_loss is None:
            boundary_loss = 0
        if labels is not None:
            ner_loss = -self.crf(
                emissions,
                labels,
                mask=ner_mask,
                reduction="mean",
            )

            loss = (
                    ner_loss
                    + self.lambda_boundary
                    * boundary_loss
            )

            if (
                    self.use_amgca
                    and self.args.use_prompt
                    and images is not None
            ):
                loss = (
                        loss
                        + self.lambda_contrastive
                        * contrastive_loss
                )
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
    def _create_entity_mask(
        self,
        boundary_tags,
        attention_mask,
    ):

        batch_size, seq_len = (
            attention_mask.shape
        )

        entity_mask = torch.zeros(
            batch_size,
            seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        for batch_index in range(batch_size):

            tags = boundary_tags[batch_index]

            if isinstance(tags, torch.Tensor):
                tags = (
                    tags.detach()
                    .cpu()
                    .tolist()
                )
            valid_length = int(
                attention_mask[
                    batch_index
                ].sum().item()
            )
            valid_length = min(
                valid_length,
                len(tags),
                seq_len,
            )

            for token_index in range(valid_length):
                tag_id = int(
                    tags[token_index]
                )
                if (
                    tag_id < 0
                    or tag_id
                    >= self.num_boundary_labels
                ):
                    raise RuntimeError(
                        f"Invalid boundary tag ID "
                        f"{tag_id} at batch "
                        f"{batch_index}, position "
                        f"{token_index}. Expected "
                        f"range [0, "
                        f"{self.num_boundary_labels - 1}]."
                    )

                if tag_id != self.boundary_o_id:
                    entity_mask[
                        batch_index,
                        token_index,
                    ] = 1
        return entity_mask

