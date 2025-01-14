import os
import time

from pathlib import Path
from screenshot import GameScreenshot
from sam_segmentation import SAMSegmentation
from vit_recognition import ViTRecognition
from game_controller import GameController
from stopping_detection import StoppingFrameCapture
from Symbol_recognition.grid_recognizer import *
from json_to_excel import Excel_parser
import cv2
from value_recognition import ValueRecognition
import threading
import queue

GAME = 'golden'

symbol_template_dir = {
    "base":Path(f'./images/{GAME}/symbols/base_game'),
    "free":Path(f'./images/{GAME}/symbols/free_game')
    }
image_dir = {
    "base":Path(f'./images/{GAME}/screenshots/base_game'),
    "free":Path(f'./images/{GAME}/screenshots/free_game')
}
save_dir = {
    "base":Path(f'./temp/{GAME}_base_output'),
    "free":Path(f'./temp/{GAME}_free_output')
}
key_frame_dir = Path(f'./temp/key_frame')

for values in (symbol_template_dir.values(), image_dir.values(), save_dir.values()):
    for dir in values:
        dir.mkdir(parents=True, exist_ok=True)
key_frame_dir.mkdir(parents=True, exist_ok=True)
value_recognize_signal = {"base":False, "free":False}

def main():
    # 初始化模組
    game_mode = 'base'
    screenshot = GameScreenshot()
    window_name = 'BlueStacks App Player'
    Snapshot = GAME
    intensity_threshold = 20
    spin_round = 2
    numerical_round_count = {"base":0, "free":0}
    global value_recognize_signal
    free_game_initialized = False

    root_dir = Path(__file__).parent.parent

    vit_model_path = os.path.join(root_dir, 'VITModel', 'vit_model.pth')
    sam_model_path = os.path.join(root_dir, 'checkpoints', 'sam2_hiera_large.pt')
    sam_model_cfg = os.path.join(root_dir, 'sam2', 'configs', 'sam2', 'sam2_hiera_l.yaml')

    images_dir = os.path.join(root_dir, 'images', Snapshot)

    sam = SAMSegmentation(Snapshot=Snapshot, images_dir=images_dir, sam2_checkpoint=sam_model_path, model_cfg=sam_model_cfg)
    valuerec = {"base":ValueRecognition(),"free":ValueRecognition()}

    # 1. 截圖
    screenshot.capture_screenshot(window_title=window_name, images_dir=images_dir, filename=Snapshot)
    
    # 2. SAM 分割
    maskDict = sam.segment_image(os.path.join(images_dir, Snapshot + ".png"))
    
    # 3. ViT 辨識
    # put your own VIT model path here
    vit = ViTRecognition(Snapshot=Snapshot, images_dir=images_dir, maskDict=maskDict,model_path=vit_model_path)
    #highest_confidence_images, template_folder = vit.classify_components()
    #vit.output_json(template_folder=os.path.join(root_dir, f"./output/{GAME}/button_recognize/"), highest_confidence_images=highest_confidence_images)
    highest_confidence_images, template_folder = vit.classify_components()
    vit.output_json(template_folder=template_folder, highest_confidence_images=highest_confidence_images)

    # 4. 操控遊戲
    screenshot.capture_screenshot(window_title=window_name, images_dir=images_dir, filename=Snapshot+'_intialshot')

    intialshot_path = os.path.join(images_dir, Snapshot+"_intialshot.png")

    intial_avg_intensities = screenshot.clickable(snapshot_path=intialshot_path,highest_confidence_images=highest_confidence_images)
    first_frame = cv2.imread(intialshot_path)
    valuerec["base"].get_board_value(intialshot_path)

    #config_file = Path(root_dir / f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
    #grid_recognizer = BaseGridRecognizer(game=GAME, mode=MODE, config_file=config_file, window_size=(1920, 1080), debug=False)


    first_frame_width = first_frame.shape[1]
    first_frame_height = first_frame.shape[0]
    grid_recognizer_config_file = Path(root_dir / f'./SlotBot_combined/Symbol_recognition/configs/{GAME}.json')
    print(grid_recognizer_config_file)
    grid_recognizer = BaseGridRecognizer(game=GAME, mode=game_mode, config_file=grid_recognizer_config_file, window_size=(first_frame_width, first_frame_height), debug=False)
    grid_recognizer.initialize_grid(first_frame)
    # temp_img = draw_grid_on_image(first_frame, grid_recognizer.grid)
    # cv2.imshow('grid', temp_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    stop_catcher = StoppingFrameCapture(window_name=window_name,save_dir=key_frame_dir, Snapshot=Snapshot, elapsed_time_threshold=3, game_name=GAME, config_file=grid_recognizer_config_file)

    def keyframes_wrapper(module_instance, key_frame_pathes, save_images):
        key_frame_pathes = stop_catcher.get_key_frames(grid=grid_recognizer.grid,intial_intensity=intial_avg_intensities,intensity_threshold=intensity_threshold,highest_confidence_images=highest_confidence_images,save_images = save_images)
        result_queue.put(key_frame_pathes)

    def profiled_keyframes_wrapper(module_instance, key_frame_pathes, save_images = True):
        # profiler = cProfile.Profile()
        # profiler.enable()
        keyframes_wrapper(module_instance, key_frame_pathes, save_images)
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.sort_stats('cumtime')  # 按累計時間排序
        # stats.print_stats(10)  

    # 使用隊列
    result_queue = queue.Queue()
    

    for i in range(spin_round):
        GameController.Windowcontrol(GameController,highest_confidence_images=highest_confidence_images, classId=10)
        print('spin round : ',i)
        start_time = time.time()

        key_frame_paths = []
        stop_catcher_thread = threading.Thread(target=profiled_keyframes_wrapper, args=(stop_catcher, key_frame_paths, True))
        stop_catcher_thread.start()

        while stop_catcher_thread.is_alive():
            time.sleep(1)
            if stop_catcher.free_gamestate:
                print('超過10秒未能恢復操作，判定已經進入免費遊戲')
        key_frame_paths = result_queue.get()

        if(stop_catcher.free_gamestate):
            game_mode="free"
        else:
            game_mode="base"

        temp_filename = Snapshot + f'_round_{i}'
        # 在base game時擷取static frame
        if(game_mode=="base"):
            stop_catcher.get_static_frame(grid=grid_recognizer.grid,images_dir=image_dir[game_mode],filename=temp_filename)
            paths = [os.path.join(image_dir[game_mode], temp_filename + '.png')]

        # 在free game或有特殊設定時使用key frame
        if(game_mode=="free"):
            paths = key_frame_paths
            numerical_round_count["free"] += len(key_frame_paths)
        elif (stop_catcher.use_key_frame):
            paths = key_frame_paths
            numerical_round_count["base"] += len(key_frame_paths)
        else:
            numerical_round_count["base"] += 1

        # 整理圖片位置與名稱
        print(f"final file name(s):")
        for j, old_path in enumerate(paths):
            new_path = os.path.join(image_dir[game_mode],temp_filename+f"-{j}.png")
            print(f"{old_path} -> {new_path}")
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
            paths[j] = new_path
        
        # 數值組10輪後，辨識每一輪數值
        if(value_recognize_signal[game_mode]):
            valuerec[game_mode].recognize_value(root_dir=root_dir, mode=GAME, image_paths=paths)
        
        # I'm not sure if this is necessary, if so, please uncomment it, otherwise you can delete this paragraph
        # if(not free_game_initialized and game_mode=="free"):
        #     valuerec["free"].get_board_value(paths[0])

        # 切換盤面辨識模式
        if grid_recognizer.mode == 'base' and game_mode=="free":
            grid_recognizer = BaseGridRecognizer(game=GAME, mode='free', config_file=grid_recognizer_config_file, window_size=(first_frame_width, first_frame_height), debug=False)
        elif grid_recognizer.mode == 'free' and game_mode=="base":
            grid_recognizer = BaseGridRecognizer(game=GAME, mode='base', config_file=grid_recognizer_config_file, window_size=(first_frame_width, first_frame_height), debug=False)

        #盤面組，每一輪建立盤面(如有需要)以及辨識盤面symbol
        for image_file in paths:
            img = cv2.imread(image_file)
            grid_recognizer.initialize_grid(img)
            grid_recognizer.recognize_roi(img)
            grid_recognizer.save_annotated_frame(img, os.path.basename(image_file).split(".")[0])
            grid_recognizer.save_grid_results(os.path.basename(image_file).split(".")[0])


        #數值組 
        for mode in ("base","free"):
            if numerical_round_count[mode] >= 10 and not value_recognize_signal[mode]:
                '''
                all_keyframes = [os.path.join(key_frame_dir, file) for file in os.listdir(key_frame_dir)]

                # Sort the files if needed (e.g., alphabetically or by modification time)
                all_keyframes.sort()

                # Collect the first `file_count` files
                all_keyframes = all_keyframes[:numerical_round_count]
                # numerical_round_cound減少1，key_frame編號記錄從0開始，round從1開始
                print('all_keyframes', all_keyframes)
                print('numerical_round_count', numerical_round_count)
                valuerec.get_meaning(all_keyframes, numerical_round_count - 1)
                value_recognize_signal = True
                '''
                all_rounds = [os.path.join(image_dir[mode], file)for file in os.listdir(image_dir[mode])]
                print('all rounds round images pathes:', all_rounds)

                valuerec[mode].get_meaning(root_dir, GAME, mode, all_rounds, numerical_round_count[mode])
                valuerec[mode].recognize_value(root_dir=root_dir, mode=GAME, image_paths=all_rounds)
                value_recognize_signal[mode] = True

        if(game_mode=="free"):
            free_game_initialized = True


if __name__ == "__main__":
    main()
    ex = Excel_parser()
    for mode in ("base","free"):
        if(value_recognize_signal[mode]):
            ex.json_to_excel(GAME, mode)
            excel_path = os.path.join(root_dir, 'excel', f'{GAME}_{mode}.xlsx')
            jsons_path = os.path.join(root_dir, f"output/{GAME}/numerical")
            ex.fill_creation_times_by_index(folder_path=jsons_path, excel_path=excel_path, output_excel=excel_path)
