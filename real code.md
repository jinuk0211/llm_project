https://github.com/unslothai/unsloth/blob/ec19e61c854dcf9104386fa63fc6c4f2944d4f35/unsloth%2Fmodels%2Fllama.py

 One of the many reasons @UnslothAI  library is fast (among many others) is its use of Grouped Query Attention (GQA). ✨

Grouped query attention aims to improve efficiency and reduce computational cost by dividing the attention heads into groups. Each group independently attends to a subset of the key and value vectors. This reduces the overall number of attention calculations needed, leading to faster processing.

Unsloth's code efficiently handles this grouping by expanding and reshaping the K and V tensors to align with the grouped structure. This allows the attention mechanism to operate on the groups independently, achieving the desired optimization.

----

Lets go through its implementation of GQA (taken from its official source code)

📌 The code first checks if the number of groups (n_groups) is not equal to 1. If it is 1, it means there's no grouping, and the standard attention mechanism is used.

📌 If grouped query attention is employed (n_groups > 1), the key (K) and value (V) tensors are expanded along a new dimension to accommodate the groups. This is achieved using the unsqueeze and expand operations:

📌 unsqueeze: Adds a new dimension of size 1 at the specified position (here, the third dimension).
expand: Replicates the tensor along the new dimension to match the desired number of groups.

📌 Reshaping: After expanding, both K and V tensors are reshaped to combine the key/value heads and group dimensions into a single dimension representing the total number of heads (n_heads). This ensures compatibility with the subsequent attention calculation steps.

----

📌 What is the variable `Knn` in the code

```
Knn = Kn[:, :, slicing_tokens:, :]
```

Within the context of the `LlamaAttention_fast_forward_inference` function, Knn represents a modified version of the key tensor (K) used for attention calculations during fast inference with KV cache. Let's break down its role and illustrate it with an example:
Background:

`KV Cache`: During fast inference, Unsloth utilizes KV caching to store key (K) and value (V) attention tensors from previous tokens. This avoids redundant computations and accelerates the process.

`Sliding Window Attention (Optional)`: Llama models can optionally employ sliding window attention, where only a recent window of keys and values is considered for attention calculations. This further improves efficiency for long sequences.

Knn is derived from the original key tensor (K) after incorporating information from the KV cache and potentially applying the sliding window mechanism.
Purpose: Knn serves as the key tensor specifically used for attention calculations in the current inference step, taking into account past context and potential windowing restrictions.

Example: Let's assume:

```py
seq_len (length of the KV cache) = 100
sliding_window = 50

```
**Initial State**: The KV cache stores keys and values for the past 100 tokens.

**New Token:** A new token is processed, and its corresponding key is added to the KV cache.

**Sliding Window Application:** Due to the sliding window, only the most recent 50 keys (from the KV cache) are relevant for attention calculations.
'Knn' Creation: Knn is created by taking a slice of the KV cache containing the last 50 keys.

In essence, Knn ensures that the attention mechanism operates on the appropriate set of keys based on the KV cache context and the potential application of the sliding window technique, optimizing the fast inference process.

--------

📌 Decoding the Sliding Window Logic

`slicing_tokens = 1 - sliding_window`

Purpose: This line calculates the starting index for slicing the key tensor (Kn) to extract the relevant portion for attention calculations within the sliding window.

The sliding window size (sliding_window) determines the number of recent tokens to consider. Subtracting this value from 1 gives us the index of the first token within the desired window (counting from the end). For example, if sliding_window is 50, slicing_tokens would be -49, indicating that we want to include the last 50 tokens (index -49 to the end).

📌 `Knn = Kn[:, :, slicing_tokens:, :]`

Purpose: This line slices the key tensor (Kn) to create Knn, which contains only the keys within the sliding window.

The slicing operation extracts a portion of the tensor based on specified indices.

Here, `[:, :, slicing_tokens:, :]` indicates:

`[:, :]` Keep all elements along the first two dimensions (batch size and number of heads).

`slicing_tokens:` Start slicing from the index calculated in the previous step (e.g., -49) and include all remaining elements up to the end.

`[:]` Keep all elements along the last dimension (head dimension).

In summary, these lines work together to select the appropriate subset of keys from the KV cache based on the sliding window size, ensuring that the attention mechanism focuses only on the relevant recent context during fast inference.

https://github.com/vikhyat/moondream

https://github.com/vllm-project/vllm

https://github.com/jshuadvd/LongRoPE