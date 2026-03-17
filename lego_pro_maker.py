def process_image(img_pil, grid_w, grid_h, brightness, contrast):
    """處理影像：支援長寬自由設定"""
    # 預處理亮度與對比
    img = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    
    # 執行非等比例縮放 (直接縮放到使用者設定的積木數)
    img_small = img.resize((grid_w, grid_h), Image.LANCZOS)
    img_np = np.array(img_small)
    
    # --- 以下保持之前的 CIELAB 色彩匹配邏輯 ---
    p_names = list(LEGO_PALETTE.keys())
    p_rgbs = np.array(list(LEGO_PALETTE.values())) / 255.0
    p_lab = color.rgb2lab(p_rgbs.reshape(-1, 1, 3)).reshape(-1, 3)
    img_lab = color.rgb2lab(img_np / 255.0)
    
    tree = KDTree(p_lab)
    _, indices = tree.query(img_lab)
    
    lego_rgb = (p_rgbs[indices] * 255).astype(np.uint8).reshape(grid_h, grid_w, 3)
    used_colors = [p_names[i] for i in indices.flatten()]
    return lego_rgb, used_colors, img_np

# --- Streamlit 介面部分 ---
with st.sidebar:
    st.header("🛠️ 尺寸設定")
    file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
    
    # 這裡改成兩個輸入框
    col_w, col_h = st.columns(2)
    with col_w:
        grid_w = st.number_input("闊度 (寬)", min_value=8, max_value=400, value=48)
    with col_h:
        grid_h = st.number_input("長度 (高)", min_value=8, max_value=400, value=48)
    
    st.info(f"📏 實體尺寸: {grid_w*0.8:.1f} x {grid_h*0.8:.1f} cm")
    
    # 增加一個「保持原圖比例」的勾選框 (選配)
    keep_aspect = st.checkbox("自動維持原圖比例", value=False)
    
    if keep_aspect and file:
        raw_img = Image.open(file)
        w, h = raw_img.size
        grid_h = int(grid_w * (h / w))
        st.write(f"已自動調整高度為: {grid_h}")