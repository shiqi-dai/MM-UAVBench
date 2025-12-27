from datetime import date
import re
import warnings

from .image_base import ImageBaseDataset
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
import pandas as pd
from tqdm import tqdm

import json
import os
from PIL import Image, ImageDraw, ImageFont

class MMUAVBench_Image(ImageBaseDataset):
    TYPE = 'MCQ'
    DATASET = {
        'Planning': ["Air_Ground_Collaborative_Planning", "Swarm_Collaborative_Planning"]
    }

    def __init__(self, dataset='UAVBench_Image'):
        ROOT = LMUDataRoot()
        self.task_root = osp.join(ROOT, 'tasks')
        self.dataset_name = dataset
        data = self.load_data(dataset)
        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                assert len(image_map[k]) > 64 
            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images] 

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
        
        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)
    @classmethod
    def supported_datasets(cls):
        return ['UAVBench_Image']
    
    def load_data(self, dataset):
        data = {
            'index': [],
            'question_id': [],
            'image_path': [],
            'question': [],
            'options': [],
            'answer': [],
            'category': [],
            'l2-category': []
        }
        global_id = 0 
        for _, task_list in self.DATASET.items():
            for task in task_list:
                task_qa = json.load(open(osp.join(self.task_root, task, ".json"), 'r', encoding='utf-8'))
                for qa in task_qa
                    data['index'].append(str(global_id))
                    global_id = global_id + 1
                    data['question_id'].append(qa["question_id"])
                    data['question'].append(qa["question"])
                    data['options'].append(qa['options'])
                    data['answer'].append(qa["answer"])
                    data['category'].append(qa["question_meta"]["category"])
                    data['l2-category'].append(qa["question_meta"]["subcategory"])
                    data['image_path'].append([osp.join(self.ROOT, item["path"]) for item in qa["metadata"]["data_resources"]])

                    ## unifiy each qa['options'] length
                    opt_num = 8 
                    for i in range(opt_num):
                        opt = chr(ord('A')+i)
                        if opt in qa['options'].keys():
                            if opt in data.keys():
                                data[opt].append(qa['options'][opt])
                            else:
                                data[opt] = [qa['options'][opt]]
                        else: # null
                            if opt in data.keys():
                                data[opt].append("")
                            else:
                                data[opt] = [""]
        return pd.DataFrame(data)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = toliststr(line['image_path'])
        question = line['question']
        options = line['options']
        options_prompt = 'Options:\n'
        for key, item in options.items():
            if item != "":
                options_prompt += f'{key}. {item}\n'
        prompt = """ You are an expert in the field of drones. Please answer the following questions based on your professional knowledge. 
You have been provided with an image and a multiple-choice question related to the image. 
Your task is to carefully analyze the input data to answer the question, choosing from the options provided. Respond with only the letter of the correct option. 
"""
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        elif tgt_path == None:
            msgs = msgs
        else: 
            msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            report_acc, mcq_vanilla_eval  
        )
        dataset = self.dataset_name
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]

        model = 'exact_matching' 
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        if model == 'exact_matching':
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )
        
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
 
        qid_map = {i: c for i, c in zip(meta['index'], meta['question_id'])}
        data['question_id'] = [qid_map[idx] for idx in data['index']]
        
        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)

        acc = report_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc
class MMUAVBench_Video(VideoBaseDataset):
    TYPE = 'Video-MCQ'
    DATASET = {
        'Cognition': ["Event_Prediction", "Event_Tracing", "Event_Understanding"]
    }
    def __init__(self, dataset='UAVBench_Video', pack=False, nframe=0, fps=3.0):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        data = self.load_data(dataset)
        self.data = data
        self.fps = fps
        self.pack = pack
        self.nframe = nframe
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'
    @classmethod
    def supported_datasets(cls):
        return ['UAVBench_Video']
    
    def load_data(self, dataset):
        data = {
            'index': [],
            'question_id': [],
            'video_path': [],
            'question': [],
            'options': [],
            'answer': [],
            'category': [],
            'l2-category': [],
            'frame_root': []
        }
        global_id = 0 
        for _, task_list in self.DATASET.items():
            for task in task_list:
                task_qa = json.load(open(osp.join(self.task_root, task, ".json"), 'r', encoding='utf-8'))
                for qa in task_qa:
                    data['index'].append(global_id) 
                    global_id = global_id + 1
                    data['question_id'].append(qa["question_id"])
                    data['video_path'].append(osp.join(self.ROOT, qa["metadata"]["data_resources"][0]["path"]))
                    data['question'].append(qa["question"])
                    data['options'].append(qa['options'])
                    data['answer'].append(qa["answer"])
                    data['category'].append(qa["question_meta"]["category"])
                    data['l2-category'].append(qa["question_meta"]["subcategory"])
                    data['frame_root'].append(osp.join(self.ROOT, "images", "annotated", task, "frames", str(qa["question_id"])))
                    
                    ## unifiy each qa['options'] length
                    opt_num = 8
                    for i in range(opt_num):
                        opt = chr(ord('A')+i)
                        if opt in qa['options'].keys():
                            if opt in data.keys():
                                data[opt].append(qa['options'][opt])
                            else:
                                data[opt] = [qa['options'][opt]]
                        else: # null
                            if opt in data.keys():
                                data[opt].append("")
                            else:
                                data[opt] = [""]
                
        return pd.DataFrame(data)
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        question = line['question']
        options = line['options']
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        # use frames
        frames = self.save_video_frames(line['video_path'], line['frame_root'])      
        prompt = """You are an expert in the field of drones. Please answer the following questions based on your professional knowledge. 
You will be provided with {} separate frames uniformly sampled from a video and a multiple-choice question related to the video. \
The frames are provided in chronological order of the video.
Your task is to carefully analyze the input data to answer the question, choosing from the options provided. Respond with only the letter of the correct option. 
"""
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        
        message = []
        for im in frames:
            message.append(dict(type='image', value=im))   
        message.append(dict(type='text', value=prompt.format(len(frames))))
           
        return message
    def frame_paths_fps(self, frame_root, num_frames):
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]
    
    def save_video_frames(self, vid_path, frame_root):
        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        total_duration = video_info['n_frames'] / video_info['fps']
        required_frames = int(total_duration * self.fps)
        step_size = video_info['fps'] / self.fps
        indices = [int(i * step_size) for i in range(required_frames)]
        frame_paths = self.frame_paths_fps(frame_root, len(indices))
        
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_dir = f"{self.EVAL_DIR}/temp_locks"
            os.makedirs(lock_dir, exist_ok=True)
            lock_path = osp.join(lock_dir, f"{osp.basename(vid_path)}.lock")
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)
        return frame_paths

   
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (
            report_acc, mcq_vanilla_eval  
        )
        dataset = self.dataset_name
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = 'exact_matching' 
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model
        if model == 'exact_matching':
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
  
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        qid_map = {i: c for i, c in zip(meta['index'], meta['question_id'])}
        data['question_id'] = [qid_map[idx] for idx in data['index']]
        
        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)

        acc = report_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc
    