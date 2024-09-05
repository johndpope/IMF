from dataclasses import dataclass, asdict, fields


models_dir = "pretrained_models/"


@dataclass
class DefaultPathsClass:
    psp_path: str = models_dir + "psp_ffhq_encode.pt"
    e4e_path: str = models_dir + "e4e_ffhq_encode.pt"
    farl_path:str = models_dir + "face_parsing.farl.lapa.main_ema_136500_jit191.pt"
    mobile_net_pth: str = models_dir + "mobilenet0.25_Final.pth"
    ir_se50_path: str = models_dir + "model_ir_se50.pth"
    stylegan_weights: str = models_dir + "stylegan2-ffhq-config-f.pt"
    stylegan_car_weights: str = models_dir + "stylegan2-car-config-f-new.pkl"
    stylegan_weights_pkl: str = models_dir + "stylegan2-ffhq-config-f.pkl"
    arcface_model_path: str = models_dir + "iresnet50-7f187506.pth"
    moco: str = models_dir + "moco_v2_800ep_pretrain.pt"
    curricular_face_path: str = models_dir + "CurricularFace_Backbone.pth"
    mtcnn: str = models_dir + "mtcnn"
    landmark: str = models_dir + "79999_iter.pth"

    def __iter__(self):
      for field in fields(self):
          yield field.name, getattr(self, field.name)


DefaultPaths = DefaultPathsClass()
