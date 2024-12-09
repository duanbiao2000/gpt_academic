# 导入更新UI的函数
from toolbox import update_ui

# 导入捕获异常、获取配置、Markdown转换的函数
from toolbox import CatchException, get_conf, markdown_convertion

# 导入预测函数
from request_llms.bridge_all import predict_no_ui_long_connection

# 导入输入剪辑函数
from crazy_functions.crazy_utils import input_clipping

# 导入看门狗函数
from crazy_functions.agent_fns.watchdog import WatchDog

# 导入阿里云ASR函数

from crazy_functions.live_audio.aliyunASR import AliyunASR
from loguru import logger

import threading
import time
import numpy as np
import json
import re

""" WatchDog 类可能是一个用于监控某些任务或资源状态的类。它可能包含一些方法，用于启动监控、检查状态、处理异常等 """

""" AliyunASR类或函数可以用于将实时音频流转换为文本，这在语音识别、语音助手、语音搜索等领域非常有用。
logger对象可以用于记录程序运行过程中的各种信息，包括错误、警告、信息等，这对于调试和监控程序非常有帮助。 """


def chatbot2history(chatbot):
    history = []
    skip_phrases = ["[ 请讲话 ]", "[ 等待GPT响应 ]", "[ 正在等您说完问题 ]"]
    for c in chatbot:
        filtered = [
            q.strip('<div class="markdown-body">')
            .strip("</div>")
            .strip("<p>")
            .strip("</p>")
            for q in c
            if not any(q.startswith(phrase) for phrase in skip_phrases)
        ]
        history.extend(filtered)
    return history


def visualize_audio(chatbot, audio_shape):
    # 如果chatbot为空，则添加默认的对话内容
    if len(chatbot) == 0:
        chatbot.append(["[ 请讲话 ]", "[ 正在等您说完问题 ]"])
    # 将chatbot的最后一个元素转换为列表
    chatbot[-1] = list(chatbot[-1])
    # 定义左右括号
    p1 = "「"
    p2 = "」"
    # 将最后一个元素中的左右括号内的内容替换为空
    chatbot[-1][-1] = re.sub(f"{p1}(.*){p2}", "", chatbot[-1][-1])
    # 在最后一个元素中添加左右括号，并在括号内添加audio_shape
    chatbot[-1][-1] += f"{p1}`{audio_shape}`{p2}"


class AsyncGptTask:
    def __init__(self) -> None:
        self.observe_future = []
        self.observe_future_chatbot_index = []

    def gpt_thread_worker(
        self, i_say, llm_kwargs, history, sys_prompt, observe_window, index
    ):
        try:
            MAX_TOKEN_ALLO = 2560
            i_say, history = input_clipping(
                i_say, history, max_token_limit=MAX_TOKEN_ALLO
            )
            gpt_say_partial = predict_no_ui_long_connection(
                inputs=i_say,
                llm_kwargs=llm_kwargs,
                history=history,
                sys_prompt=sys_prompt,
                observe_window=observe_window[index],
                console_slience=True,
            )
        except ConnectionAbortedError as token_exceed_err:
            logger.error("至少一个线程任务Token溢出而失败", token_exceed_err)
        except Exception as e:
            logger.error("至少一个线程任务意外失败", e)

    def add_async_gpt_task(
        self, i_say, chatbot_index, llm_kwargs, history, system_prompt
    ):
        self.observe_future.append([""])
        self.observe_future_chatbot_index.append(chatbot_index)
        cur_index = len(self.observe_future) - 1
        th_new = threading.Thread(
            target=self.gpt_thread_worker,
            args=(
                i_say,
                llm_kwargs,
                history,
                system_prompt,
                self.observe_future,
                cur_index,
            ),
        )
        th_new.daemon = True
        th_new.start()

    def update_chatbot(self, chatbot):
        for of, ofci in zip(self.observe_future, self.observe_future_chatbot_index):
            try:
                chatbot[ofci] = list(chatbot[ofci])
                chatbot[ofci][1] = markdown_convertion(of[0])
            except Exception:
                self.observe_future = []
                self.observe_future_chatbot_index = []
        return chatbot


class InterviewAssistant(AliyunASR):
    def __init__(self):
        self.capture_interval = 0.5  # second
        self.stop = False
        self.parsed_text = ""  # 下个句子中已经说完的部分, 由 test_on_result_chg() 写入
        self.parsed_sentence = ""  # 某段话的整个句子, 由 test_on_sentence_end() 写入
        self.buffered_sentence = ""  #
        self.audio_shape = ""  # 音频的可视化表现, 由 audio_convertion_thread() 写入
        # 创建一个线程事件，用于通知结果变化
        self.event_on_result_chg = threading.Event()
        # 创建一个线程事件，用于通知句子结束
        self.event_on_entence_end = threading.Event()
        # 创建一个线程事件，用于通知提交问题
        self.event_on_commit_question = threading.Event()

    def __del__(self):
        # 设置停止标志
        self.stop = True
        # 设置停止信息
        self.stop_msg = ""
        # 杀死commit_wd进程
        self.commit_wd.kill_dog = True
        # 杀死plugin_wd进程
        self.plugin_wd.kill_dog = True

    def init(self, chatbot):
        # 初始化音频采集线程
        self.captured_audio = np.array([])
        # 设置保留最近10秒的音频
        self.keep_latest_n_second = 10
        # 设置在暂停2秒后提交音频
        self.commit_after_pause_n_second = 2.0
        self.ready_audio_flagment = None
        self.stop = False
        self.plugin_wd = WatchDog(timeout=5, bark_fn=self.__del__, msg="程序终止")
        self.aut = threading.Thread(
            target=self.audio_convertion_thread, args=(chatbot._cookies["uuid"],)
        )
        self.aut.daemon = True
        self.aut.start()
        # th2 = threading.Thread(target=self.audio2txt_thread, args=(chatbot._cookies['uuid'],))
        # th2.daemon = True
        # th2.start()

    def no_audio_for_a_while(self):
        if len(self.buffered_sentence) < 7:  # 如果一句话小于7个字，暂不提交
            # 如果句子长度小于7个字符，开始监听。通常意味着程序会继续等待更多的音频输入，直到句子长度达到或超过7个字符。
            self.commit_wd.begin_watch()
        else:
            # 提交当前句子进行处理
            self.event_on_commit_question.set()

    def begin(self, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
        """
        启动语音识别插件的主要函数。

        参数:
        - llm_kwargs: LLM模型的参数。
        - plugin_kwargs: 插件的参数。
        - chatbot: 聊天机器人的对话历史记录。
        - history: 对话历史的内部表示。
        - system_prompt: 系统的提示信息。

        此函数负责初始化聊天机器人，启动语音识别功能，并根据用户的语音输入进行响应。
        """
        # 初始化聊天机器人界面
        self.init(chatbot)
        chatbot.append(["[ 请讲话 ]", "[ 正在等您说完问题 ]"])
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

        # 启动插件监听
        self.plugin_wd.begin_watch()

        # 创建异步GPT任务处理对象
        self.agt = AsyncGptTask()

        # 创建一个看门狗定时器，用于在没有音频输入时触发事件
        self.commit_wd = WatchDog(
            timeout=self.commit_after_pause_n_second,
            bark_fn=self.no_audio_for_a_while,
            interval=0.2,
        )
        self.commit_wd.begin_watch()

        # 主循环，等待语音输入并处理
        while not self.stop:
            self.event_on_result_chg.wait(timeout=0.25)  # 每0.25秒运行一次
            chatbot = self.agt.update_chatbot(chatbot)  # 更新chatbot对象
            history = chatbot2history(chatbot)
            yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
            self.plugin_wd.feed()

            # 处理部分语音识别结果更新
            if self.event_on_result_chg.is_set():
                # called when some words have finished
                self.event_on_result_chg.clear()
                chatbot[-1] = list(chatbot[-1])
                chatbot[-1][0] = self.buffered_sentence + self.parsed_text
                history = chatbot2history(chatbot)
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
                self.commit_wd.feed()

            # 处理完整句子结束
            if self.event_on_entence_end.is_set():
                # called when a sentence has ended
                self.event_on_entence_end.clear()
                self.parsed_text = self.parsed_sentence
                self.buffered_sentence += self.parsed_text
                chatbot[-1] = list(chatbot[-1])
                chatbot[-1][0] = self.buffered_sentence
                history = chatbot2history(chatbot)
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

            # 处理提交问题事件
            if self.event_on_commit_question.is_set():
                # called when a question should be commited
                self.event_on_commit_question.clear()
                if len(self.buffered_sentence) == 0:
                    raise RuntimeError

                self.commit_wd.begin_watch()
                chatbot[-1] = list(chatbot[-1])
                chatbot[-1] = [self.buffered_sentence, "[ 等待GPT响应 ]"]
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
                # 添加GPT任务
                history = chatbot2history(chatbot)
                self.agt.add_async_gpt_task(
                    self.buffered_sentence,
                    len(chatbot) - 1,
                    llm_kwargs,
                    history,
                    system_prompt,
                )

                self.buffered_sentence = ""
                chatbot.append(["[ 请讲话 ]", "[ 正在等您说完问题 ]"])
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

            # 如果没有事件被触发，显示音频波形
            if (
                not self.event_on_result_chg.is_set()
                and not self.event_on_entence_end.is_set()
                and not self.event_on_commit_question.is_set()
            ):
                visualize_audio(chatbot, self.audio_shape)
                yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

        # 如果停止消息不为空，抛出运行时错误
        if len(self.stop_msg) != 0:
            raise RuntimeError(self.stop_msg)


@CatchException
def 语音助手(
    txt, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request
):
    """
    语音助手函数，用于处理语音输入并生成响应。

    参数:
    txt (str): 用户输入的文本。
    llm_kwargs (dict): 语言模型的关键字参数。
    plugin_kwargs (dict): 插件的关键字参数。
    chatbot (object): 聊天机器人的界面对象。
    history (list): 聊天历史记录。
    system_prompt (str): 系统提示信息。
    user_request (str): 用户的原始请求。

    返回:
    无返回值，但会通过chatbot对象和history列表与用户进行交互。
    """
    # 初始提示信息，通知用户可以开始语音输入
    chatbot.append(
        [
            "对话助手函数插件：使用时，双手离开鼠标键盘吧",
            "音频助手, 正在听您讲话（点击“停止”键可终止程序）...",
        ]
    )
    yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面

    # 尝试导入依赖，如果缺少依赖，则给出安装建议
    try:
        import nls
        from scipy import io
    except Exception:
        chatbot.append(
            [
                "导入依赖失败",
                "使用该模块需要额外依赖, 安装方法:```pip install --upgrade aliyun-python-sdk-core==2.13.3 pyOpenSSL webrtcvad scipy git+https://github.com/aliyun/alibabacloud-nls-python-sdk.git```",
            ]
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return

    # 检查是否配置了阿里云的APPKEY
    APPKEY = get_conf("ALIYUN_APPKEY")
    if APPKEY == "":
        chatbot.append(
            [
                "导入依赖失败",
                "没有阿里云语音识别APPKEY和TOKEN, 详情见https://help.aliyun.com/document_detail/450255.html",
            ]
        )
        yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
        return

    yield from update_ui(chatbot=chatbot, history=history)  # 刷新界面
    ia = InterviewAssistant()
    # 开始语音识别和处理
    yield from ia.begin(llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
