import os

def convert_label_to_yolo(label_path):
    current_txt = None
    current_width = 0
    current_height = 0
    bboxes = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                # 保存前一个文件的标注
                if current_txt is not None and bboxes:
                    with open("datasets/WIDER_train/labels/"+current_txt, 'w') as out_f:
                        out_f.write('\n'.join(bboxes))
                
                # 解析新文件信息
                parts = line.split()
                img_path = parts[1].split('/')[-1]
                img_width = int(parts[2])
                img_height = int(parts[3])
                
                # 生成对应的txt文件名
                base_name = os.path.basename(img_path)
                txt_name = os.path.splitext(base_name)[0] + '.txt'
                
                # 重置状态
                current_txt = txt_name
                current_width = img_width
                current_height = img_height
                bboxes = []
                
            else:
                # 处理边界框数据
                if current_txt is None:
                    continue  # 跳过没有图片信息的标注
                
                coords = list(map(float, line.split()))
                # if len(coords) != 4:
                #     continue
                
                x1, y1, x2, y2 = coords[:4]
                
                # 转换为YOLO格式
                x_center = (x1 + x2) / 2 / current_width
                y_center = (y1 + y2) / 2 / current_height
                width = (x2 - x1) / current_width
                height = (y2 - y1) / current_height
                
                # 格式化并保存
                yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                bboxes.append(yolo_line)
        
        # 保存最后一个文件的标注
        if current_txt is not None and bboxes:
            with open("datasets/WIDER_train/labels/"+current_txt, 'w') as out_f:
                out_f.write('\n'.join(bboxes))

if __name__ == "__main__":
    label_file = "datasets/WIDER_train/label.txt"  # 修改为实际路径
    convert_label_to_yolo(label_file)
    print("转换完成！")
