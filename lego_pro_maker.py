import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.spatial import KDTree
from skimage import color
import cv2
from collections import Counter

# --- 1. 完整色庫與大師風格定義 ---
FULL_PALETTE = {
    "White": (255, 255, 255), "Black": (0, 0, 0), "Light Grey": (159, 161, 158),
    "Dark Grey": (100, 100, 100), "Red": (180, 0, 0), "Blue": (30, 90, 168),
    "Yellow": (250, 200, 10), "Green": (0, 133, 43), "Orange": (255, 126, 20),
    "Tan": (210, 182, 131), "Pink": (255, 187, 201), "Purple": (129, 30, 159),
    "Lime": (183, 212, 37), "Cyan": (0, 161, 211), "Brown": (88, 57, 39)
}

MASTER_STYLES = {
    "全色域 (Full Palette)": list(FULL_PALETTE.keys()),
    "梵谷星夜 (Impressionist)": ["Blue", "Cyan", "Yellow", "Lime", "Dark Grey", "Black"],
    "波普藝術 (Pop Art)": ["Red", "Blue", "Yellow", "Pink", "Cyan", "White"],
    "大師黑白 (B&W Gallery)": ["White", "Black", "Light Grey", "Dark Grey"],
    "復古暖調 (Retro Warm)": ["Brown", "Tan", "Orange", "Yellow", "Dark Grey", "White"]
}

# --- 2. 核心影像處理函數 ---

def apply_studs_effect(img_np, scale=12):
    """渲染立體 Studs (凸點) 紋理濾鏡"""
    h, w, _ = img_np.shape
    base_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    scaled = cv2.resize(base_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    stud = np.full((scale, scale), 160, dtype=np.uint8)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.38), 235, -1)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.38), 80, 1)
    texture = np.tile(stud, (h, w))
    texture = cv2.merge([texture]*3).astype(float) / 255.0
    result = (scaled.astype(float) / 255.0) * texture * 1.45
    return cv2.cvtColor(np.clip(result * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

def process_mosaic(img_pil, grid_w, grid_h, brightness, contrast, palette_dict):
    """執行縮放、色彩調整與 CIELAB 配色"""
    img = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img_small = img.resize((grid_w, grid_h), Image.LANCZOS)
    img_np = np.array(img_small)
    
    p_names = list(palette_dict.keys())
    p_rgbs = np.array(list(palette_dict.values())) / 255.0
    p_lab = color.rgb2lab(p_rgbs.reshape(-1, 1, 3)).reshape(-1, 3)
    img_lab = color.rgb2lab(img_np / 255.0)
    
    tree = KDTree(p_lab)
    _, indices = tree.query(img_lab)
    lego_rgb = (p_rgbs[indices] * 255).astype(np.uint8).reshape(grid_h, grid_w, 3)
    used_colors = [p_names[i] for i in indices.flatten()]
    return lego_rgb, used_colors, img_np

# --- 3. Streamlit 介面 ---

def main():
    st.set_page_config(layout="wide", page_title="Ultimate Lego Master")
    st.title("🧱 Ultimate LEGO® Master Studio")

    with st.sidebar:
        st.header("📏 尺寸與照片")
        file = st.file_uploader("上傳照片", type=["jpg", "png", "jpeg"])
        col_w, col_h = st.columns(2)
        grid_w = col_w.number_input("闊度 (寬)", 8, 500, 48)
        grid_h = col_h.number_input("長度 (高)", 8, 500, 48)
        st.info(f"📏 實體尺寸: {grid_w*0.8:.1f} x {grid_h*0.8:.1f} cm")
        
        st.divider()
        st.header("🎨 風格與配色")
        style_choice = st.selectbox("選擇大師配色風格", list(MASTER_STYLES.keys()))
        selected_colors = MASTER_STYLES[style_choice]
        current_pal = {name: FULL_PALETTE[name] for name in selected_colors}
        
        st.divider()
        st.header("⚙️ 進階控制")
        bright = st.slider("亮度", 0.5, 2.0, 1.0)
        cont = st.slider("對比", 0.5, 2.0, 1.1)
        use_studs = st.toggle("顯示顆粒質感 (Studs)", value=True)
        show_guide = st.checkbox("生成分區說明書 (16x16)", value=False)

    if file:
        raw_img = Image.open(file).convert("RGB")
        lego_rgb, used_colors, small_np = process_mosaic(raw_img, grid_w, grid_h, bright, cont, current_pal)
        
        m_col1, m_col2 = st.columns([3, 1])
        
        with m_col1:
            st.subheader(f"🖼️ 渲染預覽 - {style_choice}")
            if use_studs:
                render_img = apply_studs_effect(lego_rgb)
            else:
                render_img = Image.fromarray(lego_rgb).resize((grid_w*10, grid_h*10), Image.NEAREST)
            st.image(render_img, use_column_width=True)
            
            if show_guide:
                st.divider()
                st.subheader("📖 16x16 分區拼裝圖紙")
                r_num = (grid_h // 16) + (1 if grid_h % 16 != 0 else 0)
                c_num = (grid_w // 16) + (1 if grid_w % 16 != 0 else 0)
                for r in range(r_num):
                    cols = st.columns(min(c_num, 4))
                    for c in range(len(cols)):
                        block = lego_rgb[r*16:(r+1)*16, c*16:(c+1)*16]
                        if block.size > 0:
                            with cols[c]:
                                st.caption(f"區塊 R{r+1}-C{c+1}")
                                st.image(apply_studs_effect(block, scale=10))

        with m_col2:
            st.subheader("📊 零件統計清單")
            counts = Counter(used_colors)
            df = pd.DataFrame(counts.items(), columns=["顏色", "數量"]).sort_values("數量", ascending=False)
            st.table(df)
            st.metric("總零件數", f"{sum(counts.values())} pcs")
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下載清單 CSV", csv, "lego_parts.csv", "text/csv")

if __name__ == "__main__":
    main()