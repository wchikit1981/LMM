import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.spatial import KDTree
import cv2
from collections import Counter

# --- 1. 定義官方樂高完整色庫 ---
LEGO_COLORS = {
    "White": (255, 255, 255), "Black": (0, 0, 0), "Light Grey": (159, 161, 158),
    "Dark Grey": (100, 100, 100), "Red": (180, 0, 0), "Blue": (30, 90, 168),
    "Yellow": (250, 200, 10), "Green": (0, 133, 43), "Orange": (255, 126, 20),
    "Tan": (210, 182, 131), "Pink": (255, 187, 201), "Purple": (129, 30, 159),
    "Lime": (183, 212, 37), "Cyan": (0, 161, 211), "Brown": (88, 57, 39)
}

MASTER_STYLES = {
    "🌈 全色域 (Full Palette)": list(LEGO_COLORS.keys()),
    "🌌 梵谷星夜 (Impressionist)": ["Blue", "Cyan", "Yellow", "Lime", "Dark Grey", "Black"],
    "🎨 波普藝術 (Pop Art)": ["Red", "Blue", "Yellow", "Pink", "Cyan", "White"],
    "🎞️ 大師黑白 (B&W Gallery)": ["White", "Black", "Light Grey", "Dark Grey"],
    "⏳ 復古暖調 (Retro Warm)": ["Brown", "Tan", "Orange", "Yellow", "Dark Grey", "White"]
}

# --- 2. 核心處理邏輯 ---

def apply_clahe(img_np):
    """局部對比度自適應增強：救回陰影細節"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def render_studs(img_np, scale=12, overlay_nums=None, pal_names=None):
    """渲染積木質感與數字編號"""
    h, w, _ = img_np.shape
    res = cv2.resize(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    stud = np.full((scale, scale), 160, dtype=np.uint8)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.35), 230, -1)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.35), 70, 1)
    tex = np.tile(stud, (h, w))
    tex = cv2.merge([tex]*3).astype(float) / 255.0
    final = (res.astype(float) / 255.0) * tex * 1.5
    final_img = cv2.cvtColor(np.clip(final * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    if overlay_nums is not None and pal_names is not None:
        canvas = Image.fromarray(final_img)
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(canvas)
        for y in range(h):
            for x in range(w):
                idx = overlay_nums[y, x]
                bg_mean = np.mean(img_np[y, x])
                txt_color = (255, 255, 255) if bg_mean < 128 else (0, 0, 0)
                draw.text((x*scale+2, y*scale+1), str(idx + 1), fill=txt_color)
        return np.array(canvas)
    return final_img

# --- 3. UI 主介面 ---

def main():
    st.set_page_config(layout="wide", page_title="LEGO Ultimate Studio 2026")
    st.title("🧱 LEGO® Master Studio 2026")

    with st.sidebar:
        st.header("📸 1. 規格與風格")
        file = st.file_uploader("選擇照片", type=["jpg", "png", "jpeg"])
        c1, c2 = st.columns(2)
        grid_w = c1.number_input("闊度 (Studs)", 8, 400, 64)
        grid_h = c2.number_input("長度 (Studs)", 8, 400, 64)
        
        st.divider()
        style_key = st.selectbox("藝術色盤風格", list(MASTER_STYLES.keys()))
        selected_style_colors = MASTER_STYLES[style_key]
        
        # 讓用戶進一步過濾顏色
        enabled_colors = st.multiselect("手動微調可用顏色", selected_style_colors, default=selected_style_colors)
        if not enabled_colors: st.stop()
            
        use_clahe = st.toggle("開啟局部細節強化 (CLAHE)", value=True)
        use_dither = st.toggle("開啟視覺抖動 (Dithering)", value=True)
        
        st.divider()
        st.header("⚙️ 2. 畫面微調")
        bright = st.slider("亮度", 0.5, 2.0, 1.0)
        cont = st.slider("對比", 0.5, 2.5, 1.2)
        show_guide = st.checkbox("開啟 16x16 數字說明書", value=False)

    if file:
        # 影像縮放與前處理
        raw = Image.open(file).convert("RGB")
        img_p = ImageEnhance.Contrast(ImageEnhance.Brightness(raw).enhance(bright)).enhance(cont)
        img_np = np.array(img_p.resize((grid_w, grid_h), Image.LANCZOS))
        if use_clahe: img_np = apply_clahe(img_np)

        # 建立色盤 Tree
        p_rgbs = np.array([LEGO_COLORS[n] for n in enabled_colors])
        p_f = p_rgbs.astype(float) / 255.0
        tree = KDTree(p_f)

        if use_dither:
            img_f = img_np.astype(float) / 255.0
            for y in range(grid_h):
                for x in range(grid_w):
                    old_v = img_f[y, x].copy()
                    _, idx = tree.query(old_v)
                    new_v = p_f[idx]; img_f[y, x] = new_v
                    err = old_v - new_v
                    if x+1 < grid_w: img_f[y, x+1] += err * 7/16
                    if y+1 < grid_h:
                        if x>0: img_f[y+1, x-1] += err * 3/16
                        img_f[y+1, x] += err * 5/16
                        if x+1 < grid_w: img_f[y+1, x+1] += err * 1/16
            lego_rgb = (np.clip(img_f, 0, 1) * 255).astype(np.uint8)
        else:
            _, idxs = tree.query(img_np.astype(float)/255.0)
            lego_rgb = p_rgbs[idxs].astype(np.uint8).reshape(grid_h, grid_w, 3)

        # 零件計算
        _, final_idxs = tree.query(lego_rgb.astype(float)/255.0)
        final_idxs = final_idxs.reshape(grid_h, grid_w)

        # 渲染預覽
        m_col, s_col = st.columns([3, 1])
        with m_col:
            st.subheader("🖼️ 成品預覽")
            st.image(render_studs(lego_rgb, 12), use_container_width=True)
            
            if show_guide:
                st.divider()
                st.subheader("📖 16x16 數字說明書")
                rows, cols = (grid_h + 15) // 16, (grid_w + 15) // 16
                for r in range(rows):
                    ui_cols = st.columns(4)
                    for c in range(cols):
                        y_s, y_e = r*16, min((r+1)*16, grid_h)
                        x_s, x_e = c*16, min((c+1)*16, grid_w)
                        b_rgb, b_idx = lego_rgb[y_s:y_e, x_s:x_e], final_idxs[y_s:y_e, x_s:x_e]
                        
                        canvas_rgb = np.zeros((16, 16, 3), dtype=np.uint8)
                        canvas_idx = np.zeros((16, 16), dtype=int)
                        canvas_rgb[:b_rgb.shape[0], :b_rgb.shape[1]] = b_rgb
                        canvas_idx[:b_idx.shape[0], :b_idx.shape[1]] = b_idx
                        
                        with ui_cols[c % 4]:
                            st.caption(f"📍 R{r+1}-C{c+1}")
                            st.image(render_studs(canvas_rgb, 18, canvas_idx, enabled_colors), use_container_width=True)
                    st.write("")

        with s_col:
            st.subheader("📊 零件清單")
            counts = Counter(final_idxs.flatten())
            df = pd.DataFrame([{"編號": i+1, "顏色": name, "數量": counts[i]} for i, name in enumerate(enabled_colors) if counts[i]>0])
            st.table(df.sort_values("數量", ascending=False))
            st.download_button("📥 下載清單", df.to_csv(index=False).encode('utf-8-sig'), "lego_master.csv")

if __name__ == "__main__":
    main()