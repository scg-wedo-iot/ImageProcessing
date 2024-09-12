def drawImage(img_base, img_top, draw_type='center'):
    w_base = img_base.shape[1]
    h_base = img_base.shape[0]

    w_roi = img_top.shape[1]
    h_roi = img_top.shape[0]

    if draw_type == 'center':
        xc_base = int(w_base/2)
        yc_base = int(h_base/2)
        r1_base = int(yc_base - h_roi/2)
        # r2 = int(yc_base + h_roi/2)
        r2_base = r1_base + h_roi

        c1_base = int(xc_base - int(w_roi/2))
        # c2 = int(xc_base + int(w_roi/2))
        c2_base = c1_base + w_roi

    img_draw = img_base.copy()
    img_draw[r1_base:r2_base, c1_base:c2_base] = img_top

    return img_draw