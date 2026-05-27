import json
import time

import cv2
import numpy as np
import pyrealsense2 as rs

class Camera():
    """
    Camera class
    Need to be initialized with the width, height, and fps of the camera align_frames() should be called every frame !!!
    """
    def __init__(self, WIDTH, HEIGHT, fps, device_index=0):

        context = rs.context()
        devices = context.query_devices()
        selected_device_serial = devices[device_index].get_info(rs.camera_info.serial_number)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(selected_device_serial)
        self.config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, fps)

        self.config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, fps) # rs.format.rgb8
        self.profile = self.pipeline.start(self.config)



        # profile = self.pipeline.get_active_profile()
        # device = profile.get_device()
        # color_sensor = device.query_sensors()[1] 


        # color_sensor.set_option(rs.option.enable_auto_white_balance, 1)


        # auto_white_balance = False
        # if not auto_white_balance:
        #     color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        #     color_sensor.set_option(rs.option.white_balance, 4000)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to) 

        self.aligned_frames = None

        for i in range(3):
            self.align_frames()
            self.get_rgb_frame()
            time.sleep(1)


        color_intrinsics = self.intrinsics
        print("Color Camera Intrinsics:")
        print(f"fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
        print(f"ppx: {color_intrinsics.ppx}, ppy: {color_intrinsics.ppy}")
        print(f"width: {color_intrinsics.width}, height: {color_intrinsics.height}")


    def check_frames(self):
        return self.aligned_frames is not None


    def align_frames(self):
        frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(frames)
        return self.aligned_frames


    def get_rgb_frame(self):
        color_frame = self.aligned_frames.get_color_frame()
        image_np = np.asanyarray(color_frame.get_data())
        return image_np
    

    def get_depth_frame(self):
        depth_frame = self.aligned_frames.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data())
        return depth

    def pixel_to_world(self, u, v, dis):
        return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, pixel = [u, v], depth = dis)
    

    def get_pixel_to_ee(self, u, v):
        self.align_frames()
        rgb_image = self.get_rgb_frame()
        depth_image = self.get_depth_frame() 
        intrinsics = self.depth_intrinsics 
        depth_scale = self.depth_scale 


        height, width = depth_image.shape

        
        print('depth',depth_image[v, u], depth_scale,(depth_image * depth_scale).max(),(depth_image * depth_scale).min())
        z = depth_image[v, u] * depth_scale
        valid_mask = (z > 0) & (z < 5.0) 

        x = (u - intrinsics.ppx) * z / intrinsics.fx
        y = (v - intrinsics.ppy) * z / intrinsics.fy
        return x, y, z    

    @property
    def intrinsics(self):
        color_frame = self.pipeline.wait_for_frames().get_color_frame()
        return color_frame.profile.as_video_stream_profile().intrinsics
    
    @property
    def depth_intrinsics(self):
        alighed_frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = alighed_frames.get_depth_frame()
        return depth_frame.profile.as_video_stream_profile().intrinsics
    
    @property
    def depth_scale(self):
        return self.profile.get_device().first_depth_sensor().get_depth_scale()





class MultiRealSenseCapture():
    def __init__(self):
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        self.serials = [dev.get_info(rs.camera_info.serial_number) for dev in self.devices]

        if len(self.serials) < 1:
            raise RuntimeError()


        self.pipelines = {}
        for serial in self.serials:
            pipeline, config, align = self._create_pipeline(serial)
            pipeline.start(config)
            self.pipelines[serial] = {
                "pipeline":pipeline,
                "config": config,
                "align": align,
            }


    def _create_pipeline(self, serial):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        align = rs.align(rs.stream.color)
        return pipeline, config, align
    

    def get_rgbd_frames(self, with_depth : bool = True):
        results = {}
        for serial, items in self.pipelines.items():
            try:
                frames = items["pipeline"].wait_for_frames()
                aligned = items["align"].process(frames)

                depth_frame = aligned.get_depth_frame() if with_depth else None
                color_frame = aligned.get_color_frame()

                if not color_frame:
                    results[serial] = None
                    continue

                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()) if (with_depth and depth_frame) else None

                results[serial] = (rgb, depth)

            except Exception as e:
                results[serial] = None

        return results


    def show(self, with_depth:bool = True):
        prev_time = cv2.getTickCount()
        while True:
            frames = self.get_rgbd_frames(with_depth=with_depth)

            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()  
            prev_time = curr_time
            fps = 1 / time_diff if time_diff > 0 else 0 
            fps_text = f"FPS: {fps:.2f}"
            for i, (seriail, data) in enumerate(frames.items()):
                if data is None:
                    continue
                font = cv2.FONT_HERSHEY_SIMPLEX
                rgb, depth = data

                if depth is not None:
                    # depth_vis = visualize_depth(depth)
                    depth_vis = e(depth)
                    cv2.putText(depth_vis, fps_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow(f"Cam {i+1} ({seriail}) Depth", depth_vis)

                
                cv2.putText(rgb, fps_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"Cam {i+1} ({seriail}) RGB", rgb)
                
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()
        cv2.destroyAllWindows()


    def stop(self):
        for serial, items in self.pipelines.items():
            items["pipeline"].stop()


    def __del__(self):
        try:
            self.stop()
        except:
            pass


def interpolate_depth_image(depth_image: np.ndarray) -> np.ndarray:


    interpolated_image = depth_image.copy()


    height, width = interpolated_image.shape


    for i in range(1, height-1):
        for j in range(1, width-1):
            if interpolated_image[i, j] == 0:  
                
                neighbors = [
                    interpolated_image[i-1, j],   
                    interpolated_image[i+1, j],   
                    interpolated_image[i, j-1],   
                    interpolated_image[i, j+1]    
                ]
                
                valid_neighbors = [x for x in neighbors if x != 0]
                
                
                if valid_neighbors:
                    interpolated_image[i, j] = int(np.mean(valid_neighbors))
                else:
                    
                    interpolated_image[i, j] = 0

    return interpolated_image

def e(depth_image: np.ndarray, max_value: int = 400000) -> np.ndarray:

   # depth_image = interpolate_depth_image(depth_image)
    if depth_image.dtype != np.uint16:
        raise ValueError("depth_image 必须是 uint16 类型")


    max_value = depth_image.max()
    min_value = depth_image.min()
    clipped = np.clip(depth_image, min_value, max_value)
    normalized = (clipped / max_value * 255).astype(np.uint8)
    return normalized

def visualize_depth(depth):
    depth_scaled = cv2.convertScaleAbs(depth,alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_scaled,cv2.COLORMAP_JET)
    return  depth_colormap


def test_camera():
    camera = Camera(640, 480, 60)
    while True:
        camera.align_frames()

        frame = camera.get_rgb_frame()
        # bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bgr_image = frame
        #cv2.imshow("frame", bgr_image)

        depth_frame = camera.get_depth_frame()
        #cv2.imshow("frame", depth_frame)
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_TURBO)
        combined = np.hstack((bgr_image, depth_colored))
        cv2.imshow("RGB + Depth", combined)


        # camera.get_point_cloud(visualize=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def test_multicamera():
    cap = MultiRealSenseCapture()
    cap.show()

if __name__=="__main__":
    

    # test_camera()
    test_multicamera()


    