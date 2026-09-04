"""
Day41:Jack 手動校準曝光門檻用的陽春 UI。純肉眼比對用,不是要交付的功能。

v2(這次更新):改成會留歷史 —— 匯入過的圖片留在 session 裡,用 tab 在
多張圖片之間快速來回切換,另外多一個「比較」tab 把所有已匯入圖片的四個
數值排成一張表,方便一次掃視多張、抓極端值。歷史只存在這次執行的 Streamlit
process 記憶體裡(st.session_state),關掉/重啟服務就會清空 —— 這是陽春
校準工具的範圍內決定,沒有另外存成檔案做「跨次執行都留著」的持久化,
如果要那種留言告訴我。

四個數值全部重用 compute_rgb_exposure.py / compute_thermal_background.py
既有的函式,不重新刻邏輯,確保這裡看到的數字跟正式 pipeline
(summarize_and_report.py / compute_sample_quality.py)算出來的是同一套:
    median      = rgb_luminance()(RGB)或 thermal_gray_array()(thermal)算完
                  灰階/亮度後的中位數
    dark_diff   = compute_rgb_exposure.median_dark_diff() 算的 median - p1
    high_frac   = compute_rgb_exposure.high_low_frac() 算的
                  pixel >= HIGH_FRAC_THRESH(240)佔比 —— Day41 新增的指標,
                  現有 pipeline 之前只有 saturated_frac(門檻 250),沒有
                  240 這個門檻,所以是新寫的,但函式跟 saturated_frac/
                  dark_frac 用同一種寫法(np.mean(布林陣列)),只是門檻
                  數字不同
    low_frac    = 同上,pixel <= LOW_FRAC_THRESH(15)佔比,對應既有
                  dark_frac(門檻 5)的新版本

RGB 跟 thermal 都是對「整張圖」算(不是特定 bbox),因為 Jack 是拿整張
圖片肉眼校準門檻,不是針對特定物件。

啟動方式:
    cd Phase3/exposure_analysis
    streamlit run calibrate_exposure_ui.py
預設開在 http://localhost:8501,終端機也會印出網址。
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from compute_rgb_exposure import rgb_luminance, median_dark_diff, high_low_frac
from compute_thermal_background import thermal_gray_array

st.set_page_config(page_title="曝光門檻校準工具", layout="wide")
st.title("曝光門檻校準工具(Day41,內部用)")
st.caption("純肉眼校準用。四個數值直接呼叫 compute_rgb_exposure.py / compute_thermal_background.py "
           "既有函式算出,跟正式 pipeline 同一套邏輯。歷史只留在這次執行期間,重啟服務會清空。")

if "history" not in st.session_state:
    st.session_state.history = []  # [{id, name, domain, img_bytes, median, dark_diff, high_frac, low_frac}]


def compute_one(img_bytes, domain):
    from io import BytesIO
    if domain == "RGB":
        img = Image.open(BytesIO(img_bytes))
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        lum = rgb_luminance(arr)
    else:
        # thermal_gray_array 吃 file-like 物件(BytesIO 也算)跟吃路徑是
        # 同一份邏輯,不是另外寫一套讀圖方式。
        lum = thermal_gray_array(BytesIO(img_bytes))
    median, dark_diff = median_dark_diff(lum)
    high_frac, low_frac = high_low_frac(lum)
    return median, dark_diff, high_frac, low_frac


with st.sidebar:
    st.subheader("匯入圖片")
    domain = st.radio("圖片類型", ["RGB", "Thermal"], horizontal=True)
    uploads = st.file_uploader(
        "選一張或多張圖片(拖曳也可以)", type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    existing_ids = {e["id"] for e in st.session_state.history}
    for f in uploads or []:
        if f.file_id in existing_ids:
            continue  # 同一批已經處理過,避免每次 rerun 重複加入
        img_bytes = f.getvalue()
        median, dark_diff, high_frac, low_frac = compute_one(img_bytes, domain)
        st.session_state.history.append({
            "id": f.file_id, "name": f.name, "domain": domain, "img_bytes": img_bytes,
            "median": median, "dark_diff": dark_diff, "high_frac": high_frac, "low_frac": low_frac,
        })

    st.caption(f"目前歷史:{len(st.session_state.history)} 張")
    if st.button("清空歷史", disabled=not st.session_state.history):
        st.session_state.history = []
        st.rerun()

if not st.session_state.history:
    st.info("從左側匯入圖片後,這裡會出現每張圖的 tab,以及一個總覽比較表。")
else:
    tab_labels = [f"{i+1}. {e['name'][:18]}" for i, e in enumerate(st.session_state.history)] + ["📊 比較全部"]
    tabs = st.tabs(tab_labels)

    for tab, entry in zip(tabs[:-1], st.session_state.history):
        with tab:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(entry["img_bytes"], caption=f"{entry['name']}  ({entry['domain']})",
                          use_container_width=True)
            with col2:
                st.metric("median", f"{entry['median']:.2f}")
                st.metric("dark_diff (median - p1)", f"{entry['dark_diff']:.2f}")
                st.metric("high_frac (px >= 240)", f"{entry['high_frac']:.4f}")
                st.metric("low_frac (px <= 15)", f"{entry['low_frac']:.4f}")

    with tabs[-1]:
        st.markdown("所有已匯入圖片的數值總覽,方便一次掃視、排序找極端值。")
        df = pd.DataFrame([{
            "#": i + 1, "檔名": e["name"], "domain": e["domain"],
            "median": round(e["median"], 2), "dark_diff": round(e["dark_diff"], 2),
            "high_frac": round(e["high_frac"], 4), "low_frac": round(e["low_frac"], 4),
        } for i, e in enumerate(st.session_state.history)])
        st.dataframe(df, use_container_width=True, hide_index=True)
