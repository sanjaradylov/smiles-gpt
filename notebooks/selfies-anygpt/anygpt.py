"""AnyGPT: general architecture of GPT for 1D molecular modeling."""

__all__ = (
    "AnyGPTConfig",
    "AnyGPTModel",
    "AnyGPTForCausalLM",
)

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import \
    BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, \
    CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neo.modeling_gpt_neo import \
    GPTNeoMLP, GPTNeoSelfAttention


SELFATTN, POSFFN, BOTH = "a", "m", "b"
PRE_LN, POST_LN, LN_NONE = "pre-ln", "post-ln", "none"


@dataclass(repr=False)
class AnyGPTConfig(PretrainedConfig):
    """Same as ``transformers.models.gpt_neo.GPTNeoConfig`` but w/ additional
    parameters ``architecture_scheme``, ``attention_type``,
    ``add_positional_encoding``, ``layer_norm_position``.

    Parameters
    ----------
    """
    num_layers: int = 6
    architecture_scheme: Tuple[Literal[SELFATTN, POSFFN, BOTH]] = \
        (SELFATTN, SELFATTN, BOTH, BOTH, BOTH, POSFFN)
    attention_type: Literal["global", "local"] = "global"

    vocab_size: int = 240
    max_position_embeddings: int = 128
    add_positional_encoding: bool = True
    window_size: Optional[int] = None
    num_heads: int = 12
    hidden_size: int = num_heads * 32
    intermediate_size: int = hidden_size * 4
    activation_function: Literal["gelu_new", "silu", "relu"] = "gelu_new"

    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    attention_dropout: float = 0.1

    layer_norm_position: Literal[PRE_LN, POST_LN, LN_NONE] = PRE_LN
    layer_norm_epsilon: float = 1e-7

    initializer_range: float = 0.02
    scale_attn_weights: bool = True

    bos_token_id: int = 1
    eos_token_id: int = 2

    use_cache: bool = True

    kwargs: Optional[Dict[str, Any]] = None


    def __post_init__(self) -> None:
        super().__init__(bos_token_id=self.bos_token_id,
                         eos_token_id=self.eos_token_id,
                         **(self.kwargs or {}))


class AnyGPTPretrainedModel(PreTrainedModel):
    config_class = AnyGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AnyGPTAttentionOnly", "AnyGPTMLPOnly",
                         "AnyGPTAttentionAndMLP"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AnyGPTModel):
            module.gradient_checkpointing = value


class AnyGPTAttentionOnly(nn.Module):
    """Self-attn. GPT-Neo layer w/ with residual connection and optional layernorm.

    Parameters
    ----------
    config : AnyGPTConfig
        See ``AnyGPTConfig`` for more info.

        Possible options for
        * ``config.layer_norm_position``: "pre-ln", "post-ln", and "none";
        * ``config.attention_type``: "global", "local";
        * ``config.window_size``: int, if ``config.attention_type="local"``.

    Examples
    --------
    >>> import torch
    >>> config = AnyGPTConfig(vocab_size=10, num_heads=2, hidden_size=8,
    ...                       layer_norm_position="none")
    >>> attn = AnyGPTAttentionOnly(config)
    >>> assert not hasattr(attn, "ln")  # No layernorm
    >>> shape = (3, 5, config.hidden_size)
    >>> x = torch.rand(*shape)
    >>> y = attn(x)
    >>> assert y[0].size() == shape  # Verify hidden-size shape
    >>> config.layer_norm_position = "pre-ln"
    >>> assert hasattr(AnyGPTAttentionOnly(config), "ln")  # W/ layernorm
    >>> config.attention_type = "local"
    >>> config.window_size = 4
    >>> assert AnyGPTAttentionOnly(config)(x)[0].size() == shape
    """
    def __init__(self, config: AnyGPTConfig) -> None:
        super().__init__()

        self.attn = GPTNeoSelfAttention(config, config.attention_type)

        self.layer_norm_position = config.layer_norm_position
        if self.layer_norm_position != LN_NONE:
            self.ln = nn.LayerNorm(config.hidden_size,
                                   eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
            Tuple[torch.Tensor],
            Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
    ]:
        if self.layer_norm_position == PRE_LN:
            residual = self.ln(hidden_states)
        else:
            residual = hidden_states

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        if self.layer_norm_position == POST_LN:
            attn_output = self.ln(attn_output)

        hidden_states = attn_output + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class AnyGPTMLPOnly(nn.Module):
    """Positionwise 2-layer feed-forward network (1x1-conv) w/
    residual connection and optional layernorm.

    Parameters
    ----------
    config : AnyGPTConfig
        See ``AnyGPTConfig`` for more info.
        Possible options for
        * ``config.layer_norm_position``: "pre-ln", "post-ln", and "none".

    Examples
    --------
    >>> import torch
    >>> config = AnyGPTConfig(vocab_size=10, hidden_size=8, num_heads=2,
    ...                       intermediate_size=16, layer_norm_position="none")
    >>> mlp = AnyGPTMLPOnly(config)
    >>> assert not hasattr(mlp, "ln")
    >>> shape = (3, 5, config.hidden_size)
    >>> x = torch.rand(*shape)
    >>> y = mlp(x)
    >>> assert y[0].size() == shape
    >>> config.layer_norm_position = "pre-ln"
    >>> assert hasattr(AnyGPTMLPOnly(config), "ln")
    """
    def __init__(self, config: AnyGPTConfig) -> None:
        super().__init__()

        self.mlp = GPTNeoMLP(config.intermediate_size, config)

        self.layer_norm_position = config.layer_norm_position
        if self.layer_norm_position != LN_NONE:
            self.ln = nn.LayerNorm(config.hidden_size,
                                   eps=config.layer_norm_epsilon)

    def forward(self, hidden_states: Optional[torch.FloatTensor],
                *a, **kw) -> torch.FloatTensor:
        residual = hidden_states

        if self.layer_norm_position == PRE_LN:
            hidden_states = self.ln(hidden_states)

        hidden_states = self.mlp(hidden_states)

        if self.layer_norm_position == POST_LN:
            hidden_states = self.ln(hidden_states)

        return (hidden_states + residual,)


class AnyGPTAttentionAndMLP(nn.Module):
    """Self-attn. GPT-Neo layer & Positionwise MLP (1x1-conv).

    Parameters
    ----------
    config : AnyGPTConfig
        See ``AnyGPTConfig`` for more info.

        Possible options for ``config.layer_norm_position``:
        "pre-ln", "post-ln", and "none".
    attention_type : {"global", "local"}, default="global"
        Global (regular) attention or local attention w/
        window size = ``config.window_size``.
    """

    def __init__(
        self,
        config: AnyGPTConfig,
        attention_type: Literal["global", "local"] = "global",
    ) -> None:
        super().__init__()

        self.attn = AnyGPTAttentionOnly(config)
        self.mlp = AnyGPTMLPOnly(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
            Tuple[torch.Tensor],
            Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]
    ]:
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = self.mlp(attn_output)[0]

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class AnyGPTModel(AnyGPTPretrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.add_positional_encoding:
            self.wpe = nn.Embedding(config.max_position_embeddings,
                                    config.hidden_size)

        self.drop = nn.Dropout(config.embed_dropout)

        self.h = nn.ModuleList([
            self._get_block(block_type)(config)
            for block_type in config.architecture_scheme
        ])

        if config.layer_norm_position == PRE_LN:
            self.ln_f = nn.LayerNorm(config.hidden_size,
                                     eps=config.layer_norm_epsilon)

        self.post_init()

    @staticmethod
    def _get_block(block_type):
        if block_type == SELFATTN:
            return AnyGPTAttentionOnly
        if block_type == POSFFN:
            return AnyGPTMLPOnly
        if block_type == BOTH:
            return AnyGPTAttentionAndMLP

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[
            Tuple[torch.Tensor],
            BaseModelOutputWithPastAndCrossAttentions
    ]:
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = (
            output_hidden_states
            or self.config.output_hidden_states
        )
        use_cache = use_cache or self.config.use_cache
        return_dict = return_dict or self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids "
                             "or inputs_embeds")

        device = (
            input_ids.device if input_ids is not None
            else inputs_embeds.device
        )

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.-attention_mask) * torch.finfo(self.dtype).min
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = inputs_embeds

        if self.config.add_positional_encoding:
            position_embeds = self.wpe(position_ids)
            hidden_states = hidden_states + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True and isinstance(block, (AnyGPTAttentionAndMLP,
                                                        AnyGPTAttentionOnly)):
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = (
                    all_self_attentions
                    + (outputs[2 if use_cache else 1],)
                )

        if self.config.layer_norm_position == PRE_LN:
            hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents,
                                     all_hidden_states, all_self_attentions]
                         if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class AnyGPTForCausalLM(AnyGPTPretrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"lm_head.weight",
        r"h\.\d+\.attn\.attention\.bias",
    ]
    _keys_to_ignore_on_save = [r"lm_head.weight"]

    def __init__(self, config: AnyGPTModel) -> None:
        super().__init__(config)

        self.transformer = AnyGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids")
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict or self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
