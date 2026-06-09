# -*- coding: utf-8 -*-
import math
# load in basic modules
import os
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from pylsl import StreamInlet, resolve_byprop
from .utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from psychopy import visual, core
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams
# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):
    """Sinusoidal approximate sampling method.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
        freqs: list of float,
            Frequencies of each stimulus.
        phases: list of float,
            Phases of each stimulus.
        srate: int or float,
            Refresh rate of screen.
        frames: int,
            Flashing frames.
        stim_color: list,
            Color of stimu.
    Returns:
    ----------
        color: ndarray,
            (n_frames, n_elements, 3)
    """

    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3))
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".
    -author: Jieyu Wu
    -Created on: 2022-12-14
    -update log:
    Parameters
    ----------
        stim_num: int,
            Number of stimuli dots of each target.
        type: int,
            avep code.

    Returns:
    ----------
        point: ndarray,
            (stim_num, 3)
    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type-1] = [1, 1, 1]
    point = np.array(point)
    return point

# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create stimulus interface.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Config positions of stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            n_elements: int,
                Number of stimuli.
            rows: int, optional,
                Rows of keyboard.
            columns: int, optional,
                Columns of keyboard.
            stim_pos: ndarray, optional,
                Extra position matrix.
            stim_length: int,
                Length of stimulus.
            stim_width: int,
                Width of stimulus.
        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions.
        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
            # note that the origin point of the coordinate axis should be the center of your screen
            # (so the upper left corner is in Quadrant 2nd), and the larger the coordinate value,
            # the farther the actual position is from the center
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be automatically converted
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the center of each block
            first_pos = (
                np.array([self.win_size[0] / columns,
                         self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception(
                    "Too much blocks or too big the stimulus region!")
            for i in range(columns):
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that need to be set on the screen
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width

    def config_text(self, symbols=None, symbol_height=0, tex_color=[1, 1, 1]):
        """Config text stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            symbols: list of str,
                Target characters.
            symbol_height: int,
                Height of target symbol.
            tex_color: list,
                Color of target symbol.
        Raises:
            Exception: Insufficient characters.
        """

        # check number of symbols
        if (symbols is not None) and (len(symbols) >= self.n_elements):
            self.symbols = symbols
        elif self.n_elements <= 40:
            self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])
        else:
            raise Exception("Please input correct symbol list!")

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 2
        self.text_stimuli = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=tex_color,
                    units="pix",
                    height=symbol_height,
                    bold=True,
                    name=symbol,
                )
            )

    def config_response(
        self,
        symbol_text="Speller:  ",
        symbol_height=0,
        symbol_color=(1, 1, 1),
        bg_color=[-1, -1, -1],
    ):
        """Config response stimuli.
        -update log:
            2022-08-10 by Wei Zhao
        Parameters
        ----------
            symbol_text: list of str,
                Online response string.
            symbol_height: int,
                Height of response symbol.
            symbol_color: list,
                Color of response symbol.
            bg_color: list,
                Color of background symbol.
        Raises:
            Exception: Insufficient characters.
        """

        brige_length = self.win_size[0] / 2 + \
            self.stim_pos[0][0] - self.stim_length / 2
        brige_width = self.win_size[1] / 2 - \
            self.stim_pos[0][1] - self.stim_width / 2

        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 3 / 3,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),
            fillColor=bg_color,
            lineColor=[1, 1, 1],
        )

        self.res_text_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_text = ">:  "
        if symbol_height == 0:
            self.symbol_height = brige_width
        self.symbol_text = symbol_text
        self.text_response = visual.TextStim(
            win=self.win,
            text=symbol_text,
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=self.symbol_height,
            bold=True,
        )


# config visual stimuli


class VisualStim(KeyboardInterface):
    """Create various visual stimuli.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    def config_index(self, index_height=0):
        """Config index stimuli: downward triangle (Unicode: \u2BC6)
        Parameters
        ----------
            index_height: int, optional,
                Defaults to 75 pixels.
        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="\u2BC6",
            font="Arial",
            color=[1.0, 1.0, 0.0],
            colorSpace="rgb",
            units="pix",
            height=index_height,
            bold=True,
            autoLog=False,
        )

#
class SSVideoStim(VisualStim):
    def __init__(self, win, video_path, stim_duration=4.0, cue_duration=2.0, rest_duration=2.0,
                 freq=None, send_marker=False, feedback=False):
        """SSVideoStim：播放稳定频率视频刺激。"""
        self.win = win
        self.stim_duration = stim_duration
        self.cue_duration = cue_duration
        self.rest_duration = rest_duration
        self.freq = freq  # 闪烁频率（Hz），若无则连续播放
        # 提示文本刺激
        self.cue_text = visual.TextStim(win, text='请保持放松，准备开始动作想象', height=0.1, pos=(0, 0))
        # 视频刺激（初始不自动播放）
        self.video = visual.MovieStim(win, filename=video_path, loop=True, autoStart=False)
        # LSL 标记输出通道（可选）
        self.outlet = None
        if send_marker:
            info = StreamInfo('MI_Markers', 'Markers', 1, 0, 'int32')
            self.outlet = StreamOutlet(info)
        # LSL 反馈输入通道（可选）
        self.inlet = None
        self.feedback = feedback
        if feedback:
            # 等待名为 'MI_Predictions' 的流出现（可按需求修改）
            streams = resolve_streams('type', 'MI_Predictions')
            if streams:
                self.inlet = StreamInlet(streams[0])

    def run_trial(self, trial_id):
        """运行单次试次：提示->任务（视频闪烁）->休息。"""
        # 提示阶段
        self.cue_text.draw()
        self.win.flip()
        core.wait(self.cue_duration)

        # 任务阶段：播放视频并可选发送标记
        if self.outlet:
            self.outlet.push_sample([trial_id + 1])  # 发送整数标记
        # 如果需要反馈，将实时读取预测结果
        pred = None

        if self.freq:
            # 闪烁播放：交替绘制视频和黑屏
            period = 1.0 / self.freq
            num_cycles = int(self.stim_duration / period)
            for i in range(num_cycles):
                # 播放半周期视频
                self.video.play()
                t0 = core.getTime()
                while core.getTime() - t0 < period / 2.0:
                    self.video.draw()
                    self.win.flip()
                self.video.pause()
                # 显示半周期黑屏
                self.win.flip()
                core.wait(period / 2.0)
        else:
            # 连续播放视频
            self.video.play()
            t0 = core.getTime()
            while core.getTime() - t0 < self.stim_duration:
                self.video.draw()
                self.win.flip()
            self.video.stop()

        # 在线反馈展示（如有）
        if self.inlet:
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            if sample:
                pred = sample[0]
        if pred is not None:
            # 简单文本反馈（可改为图像或其它）
            feedback_text = visual.TextStim(self.win, text=f'预测结果：{pred}', pos=(0,0.5), color='yellow')
            feedback_text.draw()
            self.win.flip()
            core.wait(1.0)  # 显示反馈1秒

        # 休息阶段
        self.win.flip()  # 清屏
        core.wait(self.rest_duration)

class GetPlabel_MyTherad:
    """Start a thread that receives online results
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    inlet:
        Stream data online.
    """

    def __init__(self, inlet):
        self.inlet = inlet
        self._exit = threading.Event()

    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread."""
        self._exit.clear()
        global online_text_pos, online_symbol_text
        online_text_pos = copy(self.res_text_pos)
        online_symbol_text = copy(self.symbol_text)
        while not self._exit.is_set():
            try:
                samples, _ = self.inlet.pull_sample()
                if samples:
                    # online predict id
                    predict_id = int(samples[0]) - 1
                    online_text_pos = (
                        online_text_pos[0] + self.symbol_height / 3,
                        online_text_pos[1],
                    )
                    online_symbol_text = online_symbol_text + \
                        self.symbols[predict_id]
            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()


# basic experiment control


def paradigm(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=None,
    device_type='NeuroScan'
):
    """Passing outsied parameters to inner attributes.
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
        2022-08-03 by Shengfu Wen
        2022-12-05 by Jie Mei
    Parameters
    ----------
        bg_color: ndarray,
            Background color.
        display_time: float,
            Keyboard display time before 1st index.
        index_time: float,
            Indicator display time.
        rest_time: float, optional,
            Rest-state time.
        respond_time: float, optional,
            Feedback time during online experiment.
        image_time: float, optional,
            Image time.
        port_addr:
             Computer port.
        nrep: int,
            Num of blocks.
        mi_flag: bool,
            Flag of MI paradigm.
        lsl_source_id: str,
            Source id.
        online: bool,
            Flag of online experiment.
        device_type: str,
            See support device list in brainstim README file
    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == 'NeuroScan':
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == 'Neuracle':
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type))
    port_frame = int(0.05 * fps)

    inlet = False
    if online:
        if pdim == "ssvep" or pdim == "p300" or pdim == "con-ssvep" or pdim == "avep":
            VSObject.text_response.text = copy(VSObject.reset_res_text)
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)
            VSObject.symbol_text = copy(VSObject.reset_res_text)
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data

    if pdim == "ssvep":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "avep":
        # config experiment settings
        conditions = [{'id': i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions, nrep, name='experiment', method='random')
        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()
        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(['q'])
            if 'q' in keys:
                break
            # initialise index position
            id = int(trial['id'])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.tex_height / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            for round_i in range(VSObject.n_rep):
                # phase II: rest state
                if rest_time != 0:
                    iframe = 0
                    while iframe < int(fps * rest_time):

                        if iframe == 0 and port and online and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, 2)
                        elif iframe == 0 and port and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, id + 1)
                            print(id + 1)
                        if iframe == port_frame and port and round_i == 0:
                            port.setData(0)

                        if online:
                            VSObject.rect_response.draw()
                            VSObject.text_response.draw()
                        for text_stimulus in VSObject.text_stimuli:
                            text_stimulus.draw()
                        iframe += 1
                        win.flip()

                # phase III: target stimulating
                for sf in range(VSObject.stim_frames):
                    if sf == 0 and port and online:
                        VSObject.win.callOnFlip(port.setData, 2)
                    elif sf == 0 and port:
                        VSObject.win.callOnFlip(port.setData, round_i + 1)
                    if sf == port_frame and port:
                        port.setData(0)
                    VSObject.flash_stimuli[sf].draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = VSObject.symbol_text + \
                    VSObject.symbols[predict_id]
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3, res_text_pos[1])
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "p300":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            VSObject.back_stimuli.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 21)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 21)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.back_stimuli.draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.back_stimuli.draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating

            tmp = 0
            nonzeros_label = 0
            for round_num in range(VSObject.stim_round):
                for sf in range(VSObject.stim_frames):
                    if VSObject.roworcol_label[sf] > 0 and port:
                        VSObject.win.callOnFlip(
                            port.setData,
                            VSObject.order_index[tmp],
                        )
                        nonzeros_label = sf
                        tmp += 1

                        # time_recrod.append(time.time())
                        # T = time_recrod[-1] - time_recrod[-2]
                        # print('P3:%s毫秒' % ((T)*1000))
                    if (sf - nonzeros_label) > port_frame and port:
                        port.setData(0)

                    # for text_stimulus in VSObject.text_stimuli:
                    #     text_stimulus.draw()
                    VSObject.back_stimuli.draw()
                    VSObject.flash_stimuli[round(
                        round_num*VSObject.stim_frames + sf)].draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "mi":
        # config experiment settings
        conditions = [
            {"id": 0, "name": "left_hand"},
            {"id": 1, "name": "right_hand"},
            # {"id": 2, "name": "both_hands"},
        ]
        trials = data.TrialHandler(
            conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            VSObject.normal_left_stimuli.draw()
            VSObject.normal_right_stimuli.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            if id == 0:
                image_stimuli = [VSObject.image_left_stimuli]
                normal_stimuli = [VSObject.normal_right_stimuli]
            elif id == 1:
                image_stimuli = [VSObject.image_right_stimuli]
                normal_stimuli = [VSObject.normal_left_stimuli]
            else:
                image_stimuli = [
                    VSObject.image_left_stimuli,
                    VSObject.image_right_stimuli,
                ]
                normal_stimuli = []

            # phase I: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.normal_left_stimuli.draw()
                    VSObject.normal_right_stimuli.draw()
                    iframe += 1
                    win.flip()

            # phase II: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase III: target stimulating
            iframe = 0
            while iframe < int(fps * image_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.text_stimulus.draw()
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.normal_left_stimuli.draw()
                VSObject.normal_right_stimuli.draw()
                win.flip()

                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id

                if predict_id == 0:
                    response_stimuli = [VSObject.response_left_stimuli]
                    normal_stimuli = [VSObject.normal_right_stimuli]
                elif predict_id == 1:
                    response_stimuli = [VSObject.response_right_stimuli]
                    normal_stimuli = [VSObject.normal_left_stimuli]
                else:
                    response_stimuli = [
                        VSObject.response_left_stimuli,
                        VSObject.response_right_stimuli,
                    ]
                    normal_stimuli = []

                iframe = 0
                while iframe < int(fps * response_time):
                    for _response_stimuli in response_stimuli:
                        _response_stimuli.draw()
                    if normal_stimuli:
                        for _normal_stimuli in normal_stimuli:
                            _normal_stimuli.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "con-ssvep":
        global online_text_pos, online_symbol_text

        if inlet:
            MyTherad = GetPlabel_MyTherad(inlet)
            MyTherad.feedbackThread()

        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(
            conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                MyTherad.stop_feedbackThread()
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + \
                np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.text = online_symbol_text
                        VSObject.text_response.pos = online_text_pos
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

        if inlet:
            MyTherad.stop_feedbackThread()
