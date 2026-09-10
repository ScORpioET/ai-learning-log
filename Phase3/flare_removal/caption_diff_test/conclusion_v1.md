# Flare7K++ 去光暈前後 caption 差異測試 v1

## 任務範圍(照要求執行,不多不少)
- 從 Day40 的 `all_frames_diff.json`(15,153 張 frame,對應 158 支影片/train+val+test 三個 split)依 `diff_mean` 排序,挑 **diff 最大 10 張(high_diff 組)+ diff 最小 10 張(low_diff 組)**,共 20 張。
- 對這 20 張各自的 before(原圖)/after(Flare7K++ blend 輸出)各跑一次 RGB caption pipeline(checkpoint `best_model_rgb_full_reweight2x.pt`),各生成一句 caption。
- 沒有跑全部 157/158 支影片,也沒有改動任何既有 pipeline 或下結論說「flare removal 有效/無效」——這份是證據,決策留給 Jack。

## ⚠️ 先講在前面:對照組本身就有明顯差異,這個測試方法有雜訊

low_diff 組(diff_mean 最低到 0.07~3.17,幾乎是「同一張圖」等級的像素差異)裡,**9/10 張的 caption 結構化比對上仍然判定「有差異」**,跟 high_diff 組的 10/10 幾乎沒有差別。

追查原因:caption 生成用的是 `evaluate_val.generate_batch()`,用 `torch.multinomial` 做機率抽樣(不是 greedy decoding)。額外做的控制實驗(`sampling_noise_control.py`)證實了這一點——**拿同一張圖片、同一份 CLIP feature,只換隨機種子跑 5 次**,4 張測試圖全部是 **5/5 次生成出彼此不同的 caption**(結果見 `sampling_noise_control.json`)。

也就是說,這套 caption pipeline 本身的「同圖不同次生成」雜訊,已經大到可以完全解釋 low_diff 組的差異,跟「flare removal 真的改變了圖片內容」這件事無法區分。**因此這次 20 張的 before/after 比對結果,不能用來判斷 flare removal 有沒有影響 caption**——high_diff 組看到的差異,有可能只是抽樣雜訊,不是真正的訊號。

如果之後要重新驗證這個問題,需要先把生成方式改成 greedy decoding(或固定種子 + 多次取眾數)排除這個雜訊源,才能讓 before/after 比對有意義。這是方法論層級的問題,我沒有自己改動 pipeline 或重跑,留給 Jack 決定怎麼處理。

## 結果數字(僅供參考,不建議直接拿來下結論,原因見上)

| 組別 | 文字逐字不同 | 結構化比對判定有差異 |
|---|---|---|
| high_diff(diff_mean 119.9 ~ 155.4) | 10/10 | 10/10 |
| low_diff(diff_mean 0.07 ~ 3.17) | 10/10 | 9/10 |

兩組幾乎沒有差距(10/10 vs 9/10),跟雜訊控制實驗的結論一致。

## 補充發現:high_diff 組樣本本身不夠多樣
high_diff 前 10 張裡有 8 張來自同一支影片(`rXCGRrzyh98JMJk5v`),不是 10 支不同影片的代表樣本,如果之後要重測建議先對 `all_videos_diff.json` 依影片去重再抽樣,避免單一影片的個別特性主導結果。

## 檔案位置
- `select_samples.py` / `selected_samples.json`:挑樣本邏輯與結果(含 before/after 路徑)
- `generate_captions_before_after.py` / `captions_before_after.json`:20 張的 before/after caption 生成結果
- `sampling_noise_control.py` / `sampling_noise_control.json`:雜訊控制實驗(同圖 5 次生成)
- `compare_captions.py` / `comparison_table.csv` / `comparison_table.md`:文字比對 + 結構化比對(用 `Phase3/Day32/position_binding_accuracy.py` 的 `parse_caption(caption, "v7")`,跟 `caption_fusion/fuse_captions.py` 用的是同一套 parser)
- `build_gallery.py` / `gallery/`:20 張 before/after 左右合併圖(含 caption 字幕),供肉眼複核

## 一個路徑對不上的地方
任務描述提到「套用 `claude/rgb-thermal-merge-design.md` 裡設計好的結構化比對邏輯」,但這個 repo 裡沒有 `claude/` 資料夾也沒有這個檔名(對整個檔案系統搜尋過也沒找到)。實際能用的、邏輯相同的 parser 是 `Phase3/Day32/position_binding_accuracy.py` 的 `parse_caption()`,`Phase3/caption_fusion/fuse_captions.py` 本來就在用它做同一件事(caption 解析回 position/distance/class/count),這次直接沿用同一份,沒有另外發明一套。如果 Jack 手上有另一份設計文件在別的地方,麻煩補路徑,我可以再核對邏輯是否一致。
