from lib.colab_util import generate_video_from_obj, set_renderer, video
renderer = set_renderer()
obj_path = "results/pifuhd_final/recon/result_left_rig24_r07-sd02_256.obj"
out_img_path = "results/pifuhd_final/recon/result_left_rig24_r07-sd02_256.png"
video_path = "results/pifuhd_final/recon/result_left_rig24_r07-sd02_256.mp4"
generate_video_from_obj(obj_path, out_img_path, video_path, renderer)
