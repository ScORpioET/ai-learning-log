"""
驗證 F.scaled_dot_product_attention(..., is_causal=True) 在 query 長度(L) != key 長度(S)
（也就是有 past_len 的一般化 chunked prefill 情境）時，是否正確實作「bottom-right-aligned」
causal mask。

結論（已驗證）：不正確。只有手動建構的 causal mask，以及逐 token sequential decode
（每次 L==S==1，past 逐步累積），才是對的；is_causal=True 在 L != S 時兩者都對不上。

這是在寫 export_prefill.py 之前，設計 prefill wrapper 時做的 due-diligence 測試，
用來確認「prefill wrapper 絕對不能靠 is_causal=True 處理 L!=S 的情況，只能用
past_key_value=None（L==S，past_len=0）這條已知正確的路徑」——不是先做了天真版本、
輸出爛掉之後才回頭查出來的。
"""
import torch
import torch.nn.functional as F
torch.manual_seed(0)

B, nh, hs = 1, 2, 4
past_len = 3
new_len = 5

q = torch.randn(B, nh, new_len, hs)
k_new = torch.randn(B, nh, new_len, hs)
v_new = torch.randn(B, nh, new_len, hs)
k_past = torch.randn(B, nh, past_len, hs)
v_past = torch.randn(B, nh, past_len, hs)

k_full = torch.cat([k_past, k_new], dim=2)
v_full = torch.cat([v_past, v_new], dim=2)

# 方法 A：is_causal=True，L(query)=5 != S(key)=8，測試是不是 bottom-right aligned causal
out_causal_true = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)

# 方法 B：手動組一個正確的 attn_mask（新 token i 可以看 past 全部 + 新 token 0..i）驗證用
mask = torch.zeros(new_len, past_len + new_len, dtype=torch.bool)
for i in range(new_len):
    mask[i, :past_len + i + 1] = True
out_manual_mask = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)

diff = (out_causal_true - out_manual_mask).abs().max().item()
print('is_causal=True 的輸出 vs 手動因果 mask 的輸出，最大差異:', diff)

# 方法 C：逐 token sequential decode 當 ground truth（每次只餵 1 個新 token，past 逐步累積）
outs_seq = []
cur_k, cur_v = k_past, v_past
for i in range(new_len):
    qi = q[:, :, i:i + 1, :]
    cur_k = torch.cat([cur_k, k_new[:, :, i:i + 1, :]], dim=2)
    cur_v = torch.cat([cur_v, v_new[:, :, i:i + 1, :]], dim=2)
    oi = F.scaled_dot_product_attention(qi, cur_k, cur_v, is_causal=False)  # 單 token，看全部 key 即可（本來就對）
    outs_seq.append(oi)
out_seq = torch.cat(outs_seq, dim=2)

diff2 = (out_causal_true - out_seq).abs().max().item()
print('is_causal=True 的一次性 prefill 輸出 vs 逐 token sequential decode 輸出，最大差異:', diff2)

diff3 = (out_manual_mask - out_seq).abs().max().item()
print('手動 mask 的 prefill 輸出 vs 逐 token sequential decode 輸出，最大差異（驗證手動 mask 才是對的）:', diff3)
