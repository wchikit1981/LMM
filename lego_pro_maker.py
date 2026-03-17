import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.spatial import KDTree
from skimage import color
import cv2
from collections import Counter

# --- 1. 專業樂高官方色盤 ---
LEGO_PALETTE = {
    "White": (255, 255, 255), "Black": (0, 0, 0), "Light Grey": (159, 161, 158),
    "Dark Grey": (100, 100, 100), "Red": (180, 0, 0), "Blue": (30, 90, 168),
    "Yellow": (250, 200, 10), "Green": (0, 133, 43), "Orange": (255, 126, 20),
    "Tan": (210, 182, 131), "Pink": (255, 187, 201), "Purple": (129, 30, 159),
    "Lime": (183, 212, 37), "Cyan": (0, 161, 211), "Brown": (88, 57, 39)
}

# --- 2. 核心影像處理函數 ---

def apply_studs_effect(img_np, scale=12):
    """渲染立體 Studs (凸點) 紋理濾鏡"""
    h, w, _ = img_np.shape
    base_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    scaled = cv2.resize(base_img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    
    # 建立 Stud 模板 (12x12)
    stud = np.full((scale, scale), 160, dtype=np.uint8)
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.38), 235, -1) # 亮部
    cv2.circle(stud, (scale//2, scale//2), int(scale*0.38), 80, 1)   # 陰影邊
    
    # 加上一點點高光點
    cv2.circle(stud, (int(scale*0.35), int(scale*0.35)), 1, 255, -1)

    texture = np.tile(stud, (h, w))
    texture = cv2.merge([texture]*3).astype(float) / 255.0
    
    # 混合顏色與紋理
    result = (scaled.astype(float) / 255.0) * texture * 1.45
    return cv2.cvtColor(np.clip(result * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

def process_mosaic(img_pil, grid_w, grid_h, brightness, contrast):
    """執行縮放、色彩調整與 CIELAB 配色"""
    # 影像增強
    img = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    
    # 非等比例縮放 (直接指定長闊)
    img_small = img.resize((grid_w, grid_h), Image.LANCZOS)
    img_np = np.array(img_small)
    
    # CIELAB 顏色匹配邏輯 (更符合肉眼感官)
    p_names = list(LEGO_PALETTE.keys())
    p_rgbs = np.array(list(LEGO_PALETTE.values())) / 255.0
    p_lab = color.rgb2lab(p_rgbs.reshape(-1, 1, 3)).reshape(-1, 3)
    img_lab = color.rgb2lab(img_np / 255.0)
    
    tree = KDTree(p_lab)
    _, indices = tree.query(img_lab)
    
    lego_rgb = (p_rgbs[indices] * 255).astype(np.uint8).reshape(grid_h, grid_w, 3)
    used_colors = [p_names[i] for i in indices.flatten()]
    
    return lego_rgb, used_colors, img_np

# --- 3. Streamlit 介面 ---

def main():
    st.set_page_config(layout="wide", page_title="Lego Master Studio")
    st.title("🧱 Lego Pro Master Studio")

    # 側邊欄：強大的控制中心
    with st.sidebar:
        st.header("📏 尺寸與照片設定")
        file = st.file_uploader("1. 上傳照片", type=["jpg", "png", "jpeg"])
        
        # 長闊自由設定 (分離)
        col_w, col_h = st.columns(2)
        with col_w:
            grid_w = st.number_input("闊度 (寬)", 8, 500, 48)
        with col_h:
            grid_h = st.number_input("長度 (高)", 8, 500, 48)
        
        st.info(f"📏 實體預估: {grid_w*0.8:.1f} x {grid_h*0.8:.1f} cm")
        
        st.divider()
        st.header("🎨 影像處理")
        bright = st.slider("亮度調整", 0.1, 2.0, 1.0)
        cont = st.slider("對比調整", 0.1, 2.0, 1.1)
        
        st.divider()
        st.header("💎 顯示與進階")
        use_studs = st.toggle("顯示顆粒質感 (Studs)", value=True)
        use_3d = st.toggle("開啟 3D 浮雕層次計算", value=False)
        show_guide = st.checkbox("生成分區說明書 (16x16)", value=False)

    if file:
        raw_img = Image.open(file).convert("RGB")
        
        # 執行核心處理
        lego_rgb, used_colors, small_np = process_mosaic(raw_img, grid_w, grid_h, bright, cont)
        
        # 頁面佈局
        main_col, side_col = st.columns([3, 1])
        
        with main_col:
            st.subheader("🖼️ 拼圖渲染預覽")
            if use_studs:
                render_img = apply_studs_effect(lego_rgb)
            else:
                render_img = Image.fromarray(lego_rgb).resize((grid_w*10, grid_h*10), Image.NEAREST)
            st.image(render_img, use_column_width=True)
            
            # 分區說明書邏輯
            if show_guide:
                st.divider()
                st.subheader("📖 分區說明書 (每塊底板 16x16)")
                rows_needed = (grid_h // 16) + (1 if grid_h % 16 != 0 else 0)
                cols_needed = (grid_w // 16) + (1 if grid_w % 16 != 0 else 0)
                
                for r in range(rows_needed):
                    ui_cols = st.columns(min(cols_needed, 4))
                    for c in range(len(ui_cols)):
                        # 裁切每個 16x16 區塊
                        r_s, c_s = r*16, c*16
                        block = lego_rgb[r_s : r_s+16, c_s : c_s+16]
                        if block.size > 0:
                            with ui_cols[c]:
                                st.caption(f"板塊 R{r+1}-C{c+1}")
                                st.image(apply_studs_effect(block, scale=10))

        with side_col:
            st.subheader("📊 零件統計表")
            counts = Counter(used_colors)
            df = pd.DataFrame(counts.items(), columns=["顏色", "數量"]).sort_values("數量", ascending=False)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            st.metric("總零件數", f"{sum(counts.values())} pcs")
            
            if use_3d:
                # 根據亮度計算額外的 3D Plate 疊加
                gray = cv2.cvtColor(small_np, cv2.COLOR_RGB2GRAY)
                extra_pcs = np.sum(gray > 190)
                st.metric("3D 疊層零件", f"+{extra_pcs} pcs")
                st.caption("提示：在高光處疊加透明或白色 Plate 以增強立體感。")

            # 下載零件清單
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 導出 CSV 清單", csv, "lego_parts.csv", "text/csv")

if __name__ == "__main__":
    main()