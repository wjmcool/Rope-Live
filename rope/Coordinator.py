
global resize_delay
global mem_delay
import time
import torch
import traceback
import sys
import os
import rope.GUI as GUI
import rope.VideoManager as VM
import rope.Models as Models
from rope.Dicts import DEFAULT_DATA
try:
    from rope.GPUManager import GPUManager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False
resize_delay = 1
mem_delay = 1
def quit_app():
    vm.terminate_audio_process_tree()
    gui.destroy()
def coordinator():
    global r_frame
    global resize_delay
    global mem_delay
    if gui.get_action_length() > 0:
        action.append(gui.get_action())
    if vm.get_action_length() > 0:
        action.append(vm.get_action())
    if vm.get_frame_length() > 0:
        frame.append(vm.get_frame())
    if len(frame) > 0:
        gui.set_image(frame[0], False)
        frame.pop(0)
    if vm.get_requested_frame_length() > 0:
        r_frame.append(vm.get_requested_frame())
    if len(r_frame) > 0:
        gui.set_image(r_frame[0], True)
        r_frame = []
    if len(action) > 0:
        if action[0][0] == 'load_target_video':
            vm.load_target_video(action[0][1])
            action.pop(0)
        else:
            if action[0][0] == 'load_target_image':
                vm.load_target_image(action[0][1])
                action.pop(0)
            else:
                if action[0][0] == 'play_video':
                    vm.play_video(action[0][1])
                    action.pop(0)
                else:
                    if action[0][0] == 'get_requested_video_frame':
                        vm.get_requested_video_frame(action[0][1], marker=True)
                        action.pop(0)
                    else:
                        if action[0][0] == 'get_requested_video_frame_without_markers':
                            vm.get_requested_video_frame(action[0][1], marker=False)
                            action.pop(0)
                        else:
                            if action[0][0] == 'enable_virtualcam':
                                vm.enable_virtualcam()
                                action.pop(0)
                            else:
                                if action[0][0] == 'disable_virtualcam':
                                    vm.disable_virtualcam()
                                    action.pop(0)
                                else:
                                    if action[0][0] == 'change_webcam_resolution_and_fps':
                                        vm.change_webcam_resolution_and_fps()
                                        action.pop(0)
                                    else:
                                        if action[0][0] == 'target_faces':
                                            vm.assign_found_faces(action[0][1])
                                            action.pop(0)
                                        else:
                                            if action[0][0] == 'saved_video_path':
                                                vm.saved_video_path = action[0][1]
                                                action.pop(0)
                                            else:
                                                if action[0][0] == 'vid_qual':
                                                    vm.vid_qual = int(action[0][1])
                                                    action.pop(0)
                                                else:
                                                    if action[0][0] == 'set_stop':
                                                        vm.stop_marker = action[0][1]
                                                        action.pop(0)
                                                    else:
                                                        if action[0][0] == 'set_loop_mode':
                                                            vm.loop_enabled = action[0][1]
                                                            action.pop(0)
                                                        else:
                                                            if action[0][0] == 'perf_test':
                                                                vm.perf_test = action[0][1]
                                                                action.pop(0)
                                                            else:
                                                                if action[0][0] == 'ui_vars':
                                                                    vm.ui_data = action[0][1]
                                                                    action.pop(0)
                                                                else:
                                                                    if action[0][0] == 'control':
                                                                        vm.control = action[0][1]
                                                                        action.pop(0)
                                                                    else:
                                                                        if action[0][0] == 'parameters':
                                                                            vm.parameters = action[0][1]
                                                                            action.pop(0)
                                                                        else:
                                                                            if action[0][0] == 'parameters_face_editor':
                                                                                vm.parameters_face_editor = action[0][1]
                                                                                action.pop(0)
                                                                            else:
                                                                                if action[0][0] == 'markers':
                                                                                    vm.markers = action[0][1]
                                                                                    action.pop(0)
                                                                                else:
                                                                                    if action[0][0] == 'function':
                                                                                        eval(action[0][1])
                                                                                        action.pop(0)
                                                                                    else:
                                                                                        if action[0][0] == 'clear_mem':
                                                                                            action.pop(0)
                                                                                        else:
                                                                                            if action[0][0] == 'stop_play':
                                                                                                gui.set_player_buttons_to_inactive()
                                                                                                action.pop(0)
                                                                                            else:
                                                                                                if action[0][0] == 'set_virtual_cam_toggle_disable':
                                                                                                    gui.set_virtual_cam_toggle_disable()
                                                                                                    action.pop(0)
                                                                                                else:
                                                                                                    if action[0][0] == 'disable_record_button':
                                                                                                        gui.disable_record_button()
                                                                                                        action.pop(0)
                                                                                                    else:
                                                                                                        if action[0][0] == 'clear_faces_stop_swap':
                                                                                                            gui.clear_faces()
                                                                                                            gui.toggle_swapper(0)
                                                                                                            action.pop(0)
                                                                                                        else:
                                                                                                            if action[0][0] == 'clear_stop_enhance':
                                                                                                                gui.toggle_enhancer(0)
                                                                                                                action.pop(0)
                                                                                                            else:
                                                                                                                if action[0][0] == 'clear_stop_faces_editor':
                                                                                                                    gui.toggle_faces_editor(0)
                                                                                                                    action.pop(0)
                                                                                                                else:
                                                                                                                    if action[0][0] == 'set_slider_length':
                                                                                                                        gui.set_video_slider_length(action[0][1])
                                                                                                                        action.pop(0)
                                                                                                                    else:
                                                                                                                        if action[0][0] == 'set_slider_fps':
                                                                                                                            gui.set_video_slider_fps(action[0][1])
                                                                                                                            action.pop(0)
                                                                                                                        else:
                                                                                                                            if action[0][0] == 'update_markers_canvas':
                                                                                                                                gui.update_markers_canvas()
                                                                                                                                action.pop(0)
                                                                                                                            else:
                                                                                                                                if action[0][0] == 'set_frame_position':
                                                                                                                                    gui.video_slider.set(action[0][1])
                                                                                                                                    action.pop(0)
                                                                                                                                else:
                                                                                                                                    if action[0][0] == 'face_landmarks':
                                                                                                                                        vm.face_landmarks = action[0][1]
                                                                                                                                        action.pop(0)
                                                                                                                                    else:
                                                                                                                                        if action[0][0] == 'face_editor':
                                                                                                                                            vm.face_editor = action[0][1]
                                                                                                                                            action.pop(0)
                                                                                                                                        else:
                                                                                                                                            action.pop(0)
    if resize_delay > 100:
        gui.check_for_video_resize()
        resize_delay = 0
    else:
        resize_delay += 1
    if mem_delay > 1000:
        gui.update_vram_indicator()
        mem_delay = 0
    else:
        mem_delay += 1
    vm.process()
    gui.after(1, coordinator)
def _safe_cuda_check():
    """安全检测CUDA是否真正可用（处理有CUDA版PyTorch但无NVIDIA驱动的情况）"""
    try:
        if not torch.cuda.is_available():
            return False
        else:
            _test = torch.zeros(1, device='cuda')
            del _test
            return True
    except Exception:
        return False
def run():
    global frame
    global gui
    global r_frame
    global action
    global vm
    try:
        from rope.config import SOFTWARE_VERSION
        gpu_manager = None
        if GPU_MANAGER_AVAILABLE:
            try:
                import logging
                logging.debug('Coordinator: 初始化GPU管理器...')
                gpu_manager = GPUManager()
                if gpu_manager.auto_configure():
                    gpu_info = gpu_manager.gpu_info
                    tensor_settings = gpu_manager.get_optimal_tensor_settings()
                    device = tensor_settings['device']
                    logging.debug(f'Coordinator: GPU优化完成 - {gpu_info.name}')
                    logging.debug(f'Coordinator: 使用设备: {device}')
                    if gpu_info.compute_capability == (12, 0):
                        pass
                else:
                    device = 'cuda' if _safe_cuda_check() else 'cpu'
                    logging.debug(f'Coordinator: GPU管理器配置失败，使用设备: {device}')
            except Exception as e:
                device = 'cuda' if _safe_cuda_check() else 'cpu'
                logging.error(f'Coordinator: GPU管理器初始化失败: {e}，使用设备: {device}')
        else:
            device = 'cuda' if _safe_cuda_check() else 'cpu'
        if device == 'cuda':
            DEFAULT_DATA['ProvidersPriorityTextSelMode'] = 'CUDA'
            DEFAULT_DATA['ProvidersPriorityTextSelModes'] = ['CUDA', 'TensorRT', 'TensorRT-Engine', 'CPU']
        else:
            DEFAULT_DATA['ProvidersPriorityTextSelMode'] = 'CPU'
            DEFAULT_DATA['ProvidersPriorityTextSelModes'] = ['CPU']
        models = Models.Models(device=device)
        gui = GUI.GUI(models, software_version=SOFTWARE_VERSION)
        vm = VM.VideoManager(models)
        vm.gui = gui
        models.vm = vm
        logging.debug('Initializing Beauty Filter...')
        from rope.BeautyFilter import BeautyFilter
        vm.beauty_filter = BeautyFilter(device=models.device)
        action = []
        frame = []
        r_frame = []
        gui.initialize_gui()
        gui.protocol('WM_DELETE_WINDOW', lambda: quit_app())
        coordinator()
        gui.mainloop()
    except Exception as e:
        error_msg = f'运行时出错:\n{traceback.format_exc()}'
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror('错误', error_msg)
        except:
            pass
        sys.exit(1)