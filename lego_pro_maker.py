import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.spatial import KDTree
from skimage import color
import cv2
from collections import Counter

# --- 1. 大師色庫與風格定義 ---
FULL_PALETTE = {
    "White": (255, 255, 255), "Black": (0, 0, 0), "Light Grey": (159, 161, 158),
    "Dark Grey": (100, 100, 100), "Red": (180, 0, 0), "Blue": (30, 90, 168),
    "Yellow": (250, 200, 10), "Green": (0, 133, 43), "Orange": (255, 126, 20),
    "Tan": (210, 182, 131), "Pink": (255, 187, 201), "Purple": (129, 30, 159),
    "Lime": (183, 212, 37), "Cyan": (0, 161, 211), "Brown": (88, 57, 39)
}

MASTER_STYLES = {
    "全色域 (Full)": list(FULL_PALETTE.keys()),
    "梵谷 (Impressionist)": ["Blue", "Cyan", "Yellow", "Lime", "Dark Grey", "Black"],
    "安迪沃荷 (Pop Art)": ["Red", "Blue", "Yellow", "Pink", "Cyan", "White"],
    "大師黑白 (B&W Gallery)": ["White", "Black", "Light Grey", "Dark Grey"],
    "復古暖調 (Retro Warm)": ["Brown", "Tan", "Orange", "Yellow", "Dark Grey", "White"]
}

# --- 2. 影像處理邏輯 ---

def super_sharpen(img_np):
    """強效邊緣提取 (手動開啟)"""
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_cv = cv2.filter2D(img_cv, -1, kernel)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)

def apply_dithering(img_np, palette_rgbs):
    """視覺抖動演算法 (手動開啟)"""
    h, w, _ = img_np.shape
    img_f = img_np.astype(float) / 255.0
    pal_f = palette_rgbs.astype(float) / 255.0
    tree = KDTree(pal_f)
    for y in range(h):
        for x in range(w):
            old_val = img_f[y, x].copy()
            _, idx = tree.query(old_val)
            new_val = pal_f[idx]
            img_f[y, x] = new_val
            err = old_val - new_val
            if x + 1 < w: img_f[y, x+1] += err * 7/16
            if y + 1 < h:
                if x > 0: img_f[y+1, x-1] += err * 3/16
                img_f[y+1, x] += err * 5/16
                if x + 1 < w: img_f[y+1, x+1] += err * 1/16
    return (np.clip(img_f, 0, 1) * 255).astype(np.uint8)

def apply_studs(img_np, scale=12):
    """渲染 3D 凸點"""
    h, w, _ = img_np.shape
    scaled = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    stud = np.full((scale, scale), 160, dtype=np.uint8)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.35), 235, -1)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.35), 80, 1)
    texture = np.tile(stud, (h, w))
    texture = cv2.merge([texture]*3).astype(float) / 255.0
    res = (scaled.astype(float) / 255.0) * texture * 1.5
    return cv2.cvtColor(np.clip(res * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

# --- 3. Streamlit UI ---

def main():
    st.set_page_config(layout="wide", page_title="LEGO Pro Studio")
    st.title("🧱 LEGO® Design Studio (手動模式)")

    with st.sidebar:
        st.header("📸 1. 上傳與尺寸")
        file = st.file_uploader("選擇照片", type=["jpg", "png", "jpeg"])
        cw, ch = st.columns(2)
        grid_w = cw.number_input("闊度 (Studs)", 8, 400, 64)
        grid_h = ch.number_input("長度 (Studs)", 8, 400, 64)
        
        st.divider()
        st.header("⚙️ 2. 功能切換 (預設關閉)")
        # --- 所有功能預設設為 False ---
        auto_zoom = st.checkbox("自動聚焦中心區域", value=False)
        use_sharp = st.toggle("開啟強效銳化 (對付模糊)", value=False)
        use_dither = st.toggle("開啟視覺抖動 (增加細節)", value=False)
        
        st.divider()
        st.header("🎨 3. 風格與調色")
        style_name = st.selectbox("配色風格", list(MASTER_STYLES.keys()))
        bright = st.slider("亮度調校", 0.5, 2.0, 1.0)
        cont = st.slider("對比調校", 0.5, 2.5, 1.0)
        show_guide = st.checkbox("展開分區說明書", value=False)

    if file:
        raw_img = Image.open(file).convert("RGB")
        
        if auto_zoom:
            w_o, h_o = raw_img.size
            sz = min(w_o, h_o)
            raw_img = raw_img.crop(((w_o-sz)/2, (h_o-sz)/2, (w_o+sz)/2, (h_o+sz)/2))

        img_p = ImageEnhance.Brightness(raw_img).enhance(bright)
        img_p = ImageEnhance.Contrast(img_p).enhance(cont)
        img_small = img_p.resize((grid_w, grid_h), Image.LANCZOS)
        img_np = np.array(img_small)
        
        if use_sharp: img_np = super_sharpen(img_np)
        
        pal_names = MASTER_STYLES[style_name]
        pal_rgbs = np.array([FULL_PALETTE[n] for n in pal_names])
        
        if use_dither:
            lego_rgb = apply_dithering(img_np, pal_rgbs)
        else:
            p_lab = color.rgb2lab(pal_rgbs.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
            img_lab = color.rgb2lab(img_np / 255.0)
            _, idxs = KDTree(p_lab).query(img_lab)
            lego_rgb = pal_rgbs[idxs].astype(np.uint8).reshape(grid_h, grid_w, 3)

        # 介面渲染
        m_col, s_col = st.columns([3, 1])
        with m_col:
            st.subheader("🖼️ 樂高化預覽")
            st.image(apply_studs(lego_rgb, 12), use_container_width=True)
            
            if show_guide:
                st.divider()
                st.subheader("📖 16x16 分區圖紙 (比例修正版)")
                rows, cols = (grid_h + 15) // 16, (grid_w + 15) // 16
                for r in range(rows):
                    ui_cols = st.columns(4)
                    for c in range(cols):
                        y_s, y_e = r*16, min((r+1)*16, grid_h)
                        x_s, x_e = c*16, min((c+1)*16, grid_w)
                        block = lego_rgb[y_s:y_e, x_s:x_e]
                        if block.size > 0:
                            canvas = np.zeros((16, 16, 3), dtype=np.uint8)
                            canvas[:block.shape[0], :block.shape[1]] = block
                            with ui_cols[c % 4]:
                                st.caption(f"📍 R{r+1}-C{c+1}")
                                st.image(apply_studs(canvas, 12), use_container_width=True)
                        if (c % 4 == 3) and (c < cols - 1): ui_cols = st.columns(4)
                    st.divider()

        with s_col:
            st.subheader("📊 零件統計")
            flat_pixels = lego_rgb.reshape(-1, 3)
            tree = KDTree(pal_rgbs)
            _, idxs = tree.query(flat_pixels)
            counts = Counter([pal_names[i] for i in idxs])
            df = pd.DataFrame(counts.items(), columns=["顏色", "數量"]).sort_values("數量", ascending=False)
            st.table(df)
            st.metric("總片數", f"{len(idxs)} pcs")

if __name__ == "__main__":
    main()