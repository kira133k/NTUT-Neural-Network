import os
from PIL import Image

# 設定資料夾路徑和檔名開頭
folder_path = 'Pseudoinverse_0.3'  # 依你實際資料夾修改
filename_prefix = 'sim'  # 例如 "pattern"，依你實際檔案命名修改

# 讀取所有指定開頭的 png 檔，排序
file_list = [f for f in os.listdir(folder_path)
             if f.startswith(filename_prefix) and f.endswith('.png')]
file_list = sorted(file_list)[:12]  # 最多取12張

# 讀入12張圖片
images = [Image.open(os.path.join(folder_path, fname)) for fname in file_list]

# 取得單張圖片大小
img_width, img_height = images[0].size

# 設定合成圖的尺寸（3 列 4 行）
grid_cols, grid_rows = 3, 4
result_width = img_width * grid_cols
result_height = img_height * grid_rows

# 建立空白大圖
result_img = Image.new('RGB', (result_width, result_height), color=(255, 255, 255))

# 貼上每張圖片到對應位置
for idx, img in enumerate(images):
    row = idx // grid_cols
    col = idx % grid_cols
    x = col * img_width
    y = row * img_height
    result_img.paste(img, (x, y))

# 儲存合成結果
result_img.save('combined.png')
print('合成完成：combined.png')
