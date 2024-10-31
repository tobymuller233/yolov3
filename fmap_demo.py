import onnxruntime as ort
import numpy as np
import cv2

def load_model(onnx_path):  # load onnx model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    return session, input_name

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  
    img = np.transpose(img, (2, 0, 1))  
    img = np.expand_dims(img, axis=0)  
    return img

def infer(session, input_name, img):
    outputs = session.run(None, {input_name: img})
    return outputs

if __name__ == "__main__":
    onnx_path = 'weights/fmap_model1_SCUT_A_B_640x640.onnx'
    img_path = 'data/images/bus.jpg'
    
    # 加载模型和图像
    session, input_name = load_model(onnx_path)
    img = preprocess_image(img_path)

    # 推理并获取 feature maps
    feature_maps = infer(session, input_name, img)
    
    # 打印 feature map 的形状
    for i, fm in enumerate(feature_maps):
        print(f"Feature Map {i + 1} shape: {fm.shape}")
