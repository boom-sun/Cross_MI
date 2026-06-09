import os
import glob
import re
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description
import curryreader

# ==========================================
# 1. 用户自定义配置区域
# ==========================================

SOURCE_ROOT = r'F:\数据库'  # 原始数据总目录
BIDS_ROOT = r'F:\BIDS_Dataset'  # BIDS 输出目录

PART_CONFIG = {
    "Graz经典MI范式数据库": {
        "task_name": "graz",
        "file_pattern": "*.cdt"
    },
    "SSMVEP与MI混合范式数据库": {
        "task_name": "ssmvepmi",
        "file_pattern": "*.cdt"
    },
    "跨场景因素研究范式数据库": {
        "task_name": "hybrid",
        "file_pattern": "*.cdt"
    },
    "跨场景范式在线验证数据库": {
        "task_name": "hybridonline",
        "file_pattern": "*.cnt"   # CNT 文件
    }
}

EVENT_ID = {
    "graz": {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'rest': 4},
    "ssmvepmi": {'left_MI': 1, 'right_MI': 2, 'left_AO': 3, 'right_AO': 4},
    "hybrid": {'left_hand': 1, 'right_hand': 2},
    "hybridonline": {'left_hand': 1, 'right_hand': 2}
}

LINE_FREQ = 50

# ==========================================
# 2. 辅助函数
# ==========================================

def get_session_id(folder_name):
    if "S1" in folder_name.upper():
        return "01"
    elif "S2" in folder_name.upper():
        return "02"

# ==========================================
# 3. 主逻辑
# ==========================================
from datetime import datetime  # === 修改开始：新增导入时间模块 ===
def main():
    if not os.path.exists(BIDS_ROOT):
        os.makedirs(BIDS_ROOT)

    global_sub_counter = 1
    participants = []
    record_summary = []


    for part_folder, config in PART_CONFIG.items():
        part_path = os.path.join(SOURCE_ROOT, part_folder)
        task_name = config['task_name']
        print(f"\n>>> 正在处理部分: {part_folder} (Task: {task_name})")

        subject_dirs = [d for d in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, d))]

        for sub_dir in subject_dirs:
            sub_id = f"{global_sub_counter:02d}"
            global_sub_counter += 1
            # if global_sub_counter <=77:
            #     continue
            participants.append({"participant_id": f"sub-{sub_id}", "source": task_name})
            sub_full_path = os.path.join(part_path, sub_dir)
            session_dirs = [d for d in os.listdir(sub_full_path) if os.path.isdir(os.path.join(sub_full_path, d))]

            for ses_dir in session_dirs:
                if "S1" not in ses_dir.upper() and "S2" not in ses_dir.upper():
                    continue

                ses_id = get_session_id(ses_dir)
                ses_full_path = os.path.join(sub_full_path, ses_dir)

                search_pattern = os.path.join(ses_full_path, config['file_pattern'])
                raw_files = sorted(glob.glob(search_pattern))
                print(f"  Processing: sub-{sub_id} | ses-{ses_id} | found {len(raw_files)} files")
                for run_idx, raw_file in enumerate(raw_files, start=1):
                    fname = os.path.basename(raw_file).lower()
                    print(f"    Loading: {fname}")
                    sampling_freq = None
                    n_channels = None
                    duration = None
                    reference = "nose"  # 设置参考为鼻尖
                    # ==========================================
                    # CDT 文件处理（保持你的原逻辑）
                    # ==========================================
                    if raw_file.endswith(".cdt"):
                        curry_data = curryreader.read(raw_file, plotdata=0)
                        data = curry_data['data'].T

                        # 修复 S2 触发信号
                        if "S2" in ses_dir.upper():
                            for i in range(data.shape[1]):
                                data[-1, i] = int(data[-1, i]) & 0xFF
                        if task_name in 'graz' or task_name in 'ssmvepmi':
                            # 修复 S1 的事件映射
                            if "S1" in ses_dir.upper():
                                for i in range(len(data[0, :])):
                                    for id, l in enumerate([192, 160, 224, 144]):
                                        if data[-1, i] == l:
                                            data[-1, i] = int(id + 1)
                                    if data[-1, i] not in [192, 160, 224, 144, 1, 2, 3, 4]:
                                        data[-1, i] = 0
                        else:
                            if "S1" in ses_dir.upper():
                                for i in range(len(data[0, :])):
                                    for id, l in enumerate([192, 160]):
                                        if data[-1, i] == l:
                                            data[-1, i] = int(id + 1)
                                    if data[-1, i] not in [192, 160, 1, 2]:
                                        data[-1, i] = 0

                        sfreq = curry_data['info']['samplingfreq']
                        ch_names = curry_data['labels']
                        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                        raw = mne.io.RawArray(data, info)

                        montage = mne.channels.make_dig_montage(
                            ch_pos={ch: loc[:3] for ch, loc in zip(ch_names, curry_data['sensorpos'])},
                            coord_frame='head'
                        )
                        raw.set_montage(montage, on_missing="ignore")

                        trigger_channel = 'Trigger'
                        try:
                            events = mne.find_events(raw, stim_channel=trigger_channel, shortest_event=1, verbose=False)
                            print(f"    从通道 {trigger_channel} 提取到 {len(events)} 个事件")
                        except Exception as e:
                            print(f"    提取事件失败: {e}")
                            events = None

                    # ==========================================
                    # CNT 文件处理
                    # ==========================================
                    elif raw_file.endswith(".cnt"):
                        raw = mne.io.read_raw_cnt(raw_file, preload=True, verbose=False)
                        events, event_id_from_annotations = mne.events_from_annotations(raw)

                        code_to_label = {v: k for k, v in EVENT_ID[task_name].items()}
                        new_descriptions = []
                        for event_code in events[:, 2]:  # 遍历所有事件编码
                            if event_code in [1,2]:
                                label = code_to_label.get(event_code)
                                new_descriptions.append(label)
                        new_events = np.array([events[i, :] for i in range(len(events)) if events[i, 2] in [1,2]])
                        onset = new_events[:, 0] / raw.info['sfreq']  # 样本点转秒
                        duration = np.zeros(len(new_events))
                        new_annotations = mne.Annotations(onset=onset, duration=duration,
                                                          description=new_descriptions)
                        raw.set_annotations(new_annotations)
                        # 4. 重新从新注释中提取事件，确保使用正确的标签映射
                        events, event_id = mne.events_from_annotations(raw)
                        print(f"    从注释中提取到 {len(events)} 个事件")
                    run_id = fname[-5]
                    # ==========================================
                    # 写入 BIDS
                    # ==========================================
                    if task_name in 'graz' or task_name in 'ssmvepmi':
                        bids_path = BIDSPath(
                            subject=sub_id,
                            session=ses_id,
                            task=task_name,
                            run='0'+run_id,
                            datatype='eeg',
                            root=BIDS_ROOT
                        )
                    else:
                        b_flag = 0
                        s_name = ["graz", "ssmvep", "ssvideo", "video"]
                        for id1, s1 in enumerate(
                                ['arrow_online', 'ssmveparrow_online', 'ssvideo_online', 'video_online']):
                            if s1 in fname:
                                stim_label = s_name[id1] + 'online'
                                run_id = '1'
                                break
                            for id2, s2 in enumerate(['arrow', 'ssmveparrow', 'ssvideo', 'video']):
                                if s2 in fname:
                                    stim_label = s_name[id2]
                                    break
                                    b_flag = 1
                            for id3, s3 in enumerate(["cue", "ssmvep", "ssvideo", "video"]):
                                if s3 in fname:
                                    stim_label = s_name[id3]
                                    break
                                    b_flag = 1
                            if b_flag == 1:
                                break
                        bids_path = BIDSPath(
                            subject=sub_id,
                            session=ses_id,
                            task=task_name,
                            acquisition=stim_label,
                            run='0'+run_id,
                            datatype='eeg',
                            root=BIDS_ROOT
                        )
                    # write_raw_bids(
                    #     raw,
                    #     bids_path,
                    #     events=events,
                    #     event_id=EVENT_ID[task_name],
                    #     allow_preload=True,
                    #     format='BrainVision',
                    #     overwrite=True,
                    #     verbose=False
                    # )
                    sampling_freq = raw.info['sfreq']
                    n_channels = len(raw.info['ch_names'])
                    duration = raw.times[-1]
                    trial_count = len(events) if events is not None else 0
                    event_types = list(EVENT_ID[task_name].keys())

                    record_summary.append({
                        "subject": f"sub-{sub_id}",
                        "session": f"ses-{ses_id}",
                        "task": task_name,
                        "acquisition": bids_path.acquisition if bids_path.acquisition else "N/A",
                        "run": run_id,
                        "trial_count": trial_count,
                        "sampling_freq": sampling_freq,
                        "n_channels": n_channels,
                        "duration_sec": round(duration, 2),
                        "line_freq": LINE_FREQ,
                        "reference": reference,
                        "event_types": ",".join(event_types),
                        "file": os.path.basename(bids_path.basename)
                    })

    # ==========================================
    # 生成 dataset_description.json 和 participants.tsv
    # ==========================================
    make_dataset_description(
        path=BIDS_ROOT,
        name="Cross-Sence Motor Imagery Hybird-Paradigm EEG Dataset",
        authors=["Sun Xinwei"],
        acknowledgements="Data merged from Graz MI, SSMVEP-MI, Hybrid, and Hybrid-Online paradigms.",
        funding="Supported by related research project",
        references_and_links="https://bids.neuroimaging.io/",
        doi="Unpublished"
    )

    df = pd.DataFrame(participants)
    df.to_csv(os.path.join(BIDS_ROOT, "participants.tsv"), sep='\t', index=False)
    df_rec = pd.DataFrame(record_summary)
    df_rec.to_csv(os.path.join(BIDS_ROOT, "recordings.tsv"), sep='\t', index=False)
    # ==========================================
    # 自动生成 README.md（包含详细记录）
    # ==========================================
    readme_path = os.path.join(BIDS_ROOT, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"# Cross-paradigm Motor Imagery & SSVEP EEG Dataset\n\n")
        f.write(f"Generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `bids_chage.py`.\n\n")
        f.write("## Recording Summary\n")
        f.write(
            "| Subject | Session | Task | Acquisition | Run | Trials | Fs (Hz) | Ch | Duration (s) | Line Freq | Ref | Events | File |\n")
        f.write(
            "|----------|----------|------|--------------|------|----------|---------|------|--------------|-----------|------|---------|------|\n")
        for rec in record_summary:
            f.write(f"| {rec['subject']} | {rec['session']} | {rec['task']} | {rec['acquisition']} | "
                    f"{rec['run']} | {rec['trial_count']} | {rec['sampling_freq']} | {rec['n_channels']} | "
                    f"{rec['duration_sec']} | {rec['line_freq']} | {rec['reference']} | {rec['event_types']} | "
                    f"{rec['file']} |\n")
        f.write("\n## Notes\n")
        f.write("- Reference electrode: **nose**\n")
        f.write("- CDT files corrected for triggers; CNT files read natively.\n")
        f.write("- Data exported in BrainVision `.vhdr/.eeg/.vmrk` format under BIDS.\n")
    print("\n✅ 所有数据成功转换为统一 BIDS 数据集，并生成 README.md！")

# ==========================================
# 执行入口
# ==========================================
if __name__ == '__main__':
    main()
