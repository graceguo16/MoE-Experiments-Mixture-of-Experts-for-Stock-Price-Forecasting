import torch
from transformers import AutoModelForCausalLM

# 1) 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device =", device)

# 2) 构造上下文
B, context_length, prediction_length = 2, 12, 6
seqs = torch.randn(B, context_length, device=device)

# 3) 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Maple728/TimeMoE-50M",
    device_map=device,           # "cpu" 或 "cuda"
    trust_remote_code=True,
    # 如果之后装好了 flash-attn，可加：
    # attn_implementation="flash_attention_2",
)


mean = seqs.mean(dim=-1, keepdim=True)
std  = seqs.std(dim=-1, keepdim=True).clamp_min(1e-6)
normed = (seqs - mean) / std

out = model.generate(normed, max_new_tokens=prediction_length)   
normed_pred = out[:, -prediction_length:]                        
pred = normed_pred * std + mean                                  


print("output shape     :", tuple(out.shape))
print("predictions shape:", tuple(pred.shape))
print("sample prediction:", pred[0].detach().cpu().numpy())
