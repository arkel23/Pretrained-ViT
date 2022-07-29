"""
https://github.com/google-research/electra/blob/master/flops_computation.py
Computes the flops needed for training/running transformer networks.

# We checked this code with TensorFlow"s FLOPs counting, although we had to
# correct for this issue: https://github.com/tensorflow/tensorflow/issues/22071
# Assumptions going into the FLOPs counting
#   - An "operation" is a mathematical operation, not a machine instruction. So
#     an "exp" takes one opp like and add, even though in practice an exp
#     might be slower. This is not too bad an assumption because
#     matrix-multiplies dominate the compute for most models, so minor details
#     about activation functions don"t matter too much. Similarly, we count
#     matrix-multiplies as 2*m*n flops instead of m*n, as one might if
#     if considering fused multiply-add ops.
#   - Backward pass takes the same number of FLOPs as forward pass. No exactly
#     right (e.g., for softmax cross entropy loss the backward pass is faster).
#     Importantly, it really is the same for matrix-multiplies, which is most of
#     the compute anyway.
#   - We assume "dense" embedding lookups (i.e., multiplication by a one-hot
#     vector). On some hardware accelerators, these dense operations are
#     actually faster than sparse lookups.
# Please open a github issue if you spot a problem with this code!

# I am not sure if the below constants are 100% right, but they are only applied
# to O(hidden_size) activations, which is generally a lot less compute than the
# matrix-multiplies, which are O(hidden_size^2), so they don't affect the total
# number of FLOPs much.
"""


from pretrained_vit import ViTConfig


# random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

# compute mean activation (sum), computate variance of activation
# (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5


class TransformerFLOPS(object):
    """Computes the train/inference FLOPs for transformers."""

    def __init__(self, seq_len=196, hidden_size=768, num_hidden_layers=12,
                 num_classes=1000, num_channels=3,
                 fh=16, image_size=224, intermediate_size=None, num_attention_heads=None,
                 head_size=None, decoder=False, **kwargs):
        self.h = hidden_size  # hidden size
        self.layers = num_hidden_layers  # number of layers
        self.v = num_classes  # vocab size
        self.c_in = num_channels  # num_channels
        self.fh = fh
        self.image_size = image_size
        self.s = seq_len
        self.i = hidden_size * 4 if intermediate_size is None else intermediate_size  # intermediate size
        self.kqv = hidden_size if head_size is None else head_size * num_attention_heads  # attn proj sizes
        # attention heads
        self.heads = max(hidden_size // 64, 1) if num_attention_heads is None else num_attention_heads
        self.decoder = decoder  # decoder has extra attn to encoder states

    def get_block_flops(self):
        """Get the forward-pass FLOPs for a single transformer block."""
        attn_mul = 2 if self.decoder else 1
        block_flops = dict(
            kqv=3 * 2 * self.h * self.kqv * attn_mul,
            kqv_bias=3 * self.kqv * attn_mul,
            attention_scores=2 * self.kqv * self.s * attn_mul,
            attn_softmax=SOFTMAX_FLOPS * self.s * self.heads * attn_mul,
            attention_dropout=DROPOUT_FLOPS * self.s * self.heads * attn_mul,
            attention_scale=self.s * self.heads * attn_mul,
            attention_weighted_avg_values=2 * self.h * self.s * attn_mul,
            attn_output=2 * self.h * self.h * attn_mul,
            attn_output_bias=self.h * attn_mul,
            attn_output_dropout=DROPOUT_FLOPS * self.h * attn_mul,
            attn_output_residual=self.h * attn_mul,
            attn_output_layer_norm=LAYER_NORM_FLOPS * attn_mul,
            intermediate=2 * self.h * self.i,
            intermediate_act=ACTIVATION_FLOPS * self.i,
            intermediate_bias=self.i,
            output=2 * self.h * self.i,
            output_bias=self.h,
            output_dropout=DROPOUT_FLOPS * self.h,
            output_residual=self.h,
            output_layer_norm=LAYER_NORM_FLOPS * self.h,
        )
        return sum(block_flops.values()) * self.s

    def get_patching_flops(self):
        """Get the forward-pass FLOPs for the patch projection."""
        patching_flops = {}
        patching_flops["conv"] = 2 * (self.image_size ** 2) * ((self.c_in * self.fh) + 1) * self.h
        return sum(patching_flops.values())

    def get_output_flops(self):
        output_flops = {}
        # output softmax
        output_flops.update(dict(
            hidden_layernorm=LAYER_NORM_FLOPS * self.h,
            output_softmax=SOFTMAX_FLOPS * self.v,
            linear=(2 * self.h - 1) * self.v,
        ))
        return sum(output_flops.values())

    def get_train_flops(self, batch_size, train_steps):
        """Get the FLOPs for pre-training the transformer."""
        # 2* for forward/backward pass
        return 2 * batch_size * train_steps * (self.get_infer_flops)

    def get_infer_flops(self):
        """Get the FLOPs for running inference with the transformer on a
        classification task."""
        return (
            (self.layers * self.get_block_flops()) +
            self.get_patching_flops() +
            self.get_output_flops()
        )


def get_flops(cfg):
    flops = TransformerFLOPS(**vars(cfg))
    flops = flops.get_infer_flops()
    return flops


def main():

    models = ('vit_t16', 'vit_b16', 'vit_b32', 'vit_l16')
    for name in models:
        cfg = ViTConfig(model_name=name)
        flops = get_flops(cfg)
        print('{}: {:.2f} GFLOPs'.format(name, (flops / (10 ** 9))))


if __name__ == "__main__":
    main()
