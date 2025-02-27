import torch
from utils.downloads import attempt_download
from models.common import DWConv, Conv
m1 = attempt_download("weights/model_plus_final.pt")
m1 = torch.load(m1, map_location="cpu")
# m2 = torch.load("toy/neumeta_test/ninr_yoloface500kp-1500e-coordnoise-largers-resmlp-lrstep3_1.0.pth")
# m2 = torch.load("toy/neumeta_test/gen_ninr_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_0.005.pth")
# m2 = torch.load("toy/neumeta_test/gen_ninr_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_0.5.pth")
# m2 = torch.load("toy/neumeta_test/gen_ninr_v2_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_0.5.pth")
m2 = torch.load("toy/neumeta_test/gen_ninr_v2_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_0.1.pth")
# m2 = torch.load("toy/model_plus_final.pt")
# copy the state dict of m2 to m1
m2_state_dict = {key[len("model."):]: value for key, value in m2.items()}
# m2_state_dict = m2["model"].model.state_dict()
hidden_dim = int(240 * 0.1)
m1["model"].model[21][0].cv1 = Conv(m1["model"].model[21][0].cv1.conv.in_channels, hidden_dim, 1, 1)
m1["model"].model[21][0].cv2 = DWConv(hidden_dim, hidden_dim, 3, 1)
m1["model"].model[21][0].cv3 = Conv(hidden_dim, m1["model"].model[21][0].cv3.conv.out_channels, 1, 1)
m1["model"].model[21][1].cv1 = Conv(m1["model"].model[21][1].cv1.conv.in_channels, hidden_dim, 1, 1)
m1["model"].model[21][1].cv2 = DWConv(hidden_dim, hidden_dim, 3, 1)
m1["model"].model[21][1].cv3 = Conv(hidden_dim, m1["model"].model[21][1].cv3.conv.out_channels, 1, 1)
m1["model"].model.load_state_dict(m2_state_dict)

# torch.save(m1["model"].model.state_dict(), "toy/model_plus_final.pth")
torch.save(m1, "toy/model_plus_final_dim24_v2.pt")