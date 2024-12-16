import gi
import onnxruntime as ort
from gi.repository import Gst, GObject
from huggingface_hub import hf_hub_download

Gst.init(None)

# Tải mô hình ONNX từ Hugging Face
local_onnx_path = hf_hub_download(repo_id="microsoft/Phi-3-vision-128k-instruct-onnx", filename="model.onnx")
print("ONNX model downloaded to:", local_onnx_path)

session = ort.InferenceSession(local_onnx_path)

# Tạo pipeline GStreamer
def create_pipeline(camera_source, config_file_path):
    pipeline = Gst.Pipeline.new("pipeline")
    source = Gst.ElementFactory.make(camera_source, "video-source")
    if not source:
        print(f"Không thể tạo nguồn video từ {camera_source}.")
        return None
    if camera_source == "v4l2src":
        source.set_property("device", "/dev/video0")  # Webcam
    elif camera_source == "filesrc":
        source.set_property("location", "video.mp4")  # Video file

    convert = Gst.ElementFactory.make("nvvideoconvert", "converter")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinfer")
    pgie.set_property('config-file-path', config_file_path)
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    pipeline.add(source)
    pipeline.add(convert)
    pipeline.add(pgie)
    pipeline.add(sink)

    source.link(convert)
    convert.link(pgie)
    pgie.link(sink)

    return pipeline

# Chạy pipeline
def run_pipeline(pipeline):
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline đang chạy...")
    loop = GObject.MainLoop()
    loop.run()

if __name__ == "__main__":
    camera_source = "/dev/video0"
    config_file_path = "config_file.txt"
    pipeline = create_pipeline(camera_source, config_file_path)
    if pipeline:
        run_pipeline(pipeline)
    else:
        print("Pipeline không thể khởi tạo.")
